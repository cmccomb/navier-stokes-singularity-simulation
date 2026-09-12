#ifndef NS_PAPER_FIELDS_H
#define NS_PAPER_FIELDS_H
#include "paper_profile.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <limits>

namespace ns_paper {
using namespace amrex;
inline Real phase_limit(Profile const& p,Real t) {
    if (!(t<p.T)) Abort("paper clock requires t < t_star");
    return p.phase_rate>0 ? -(p.T-t)*std::expm1(-p.phase_step/p.phase_rate) : std::numeric_limits<Real>::infinity();
}
inline FArrayBox velocity(Profile const& p,Box const& box,Geometry const& g,Real t) {
    FArrayBox result(box,3); result.setVal(0);
    if (t<=p.cutoff_start) return result;
    auto dx=g.CellSizeArray(); auto lo=g.ProbLoArray();
    if (dx[0]!=dx[1] || dx[0]!=dx[2]) Abort("paper finite differences require cubic cells");
    int passes=p.correction_strength>0 ? p.correction_passes : 0;
    Box grown=grow(box,passes+1);
    FArrayBox background(grown,3),pulse(grown,3);
    auto b=background.array(); auto a=pulse.array();
    LoopOnCpu(grown,[&](int i,int j,int k) {
        auto pot=p.potentials(lo[0]+(i+.5)*dx[0],lo[1]+(j+.5)*dx[1],lo[2]+(k+.5)*dx[2],t,dx[0]);
        for (int c=0;c<3;++c) { b(i,j,k,c)=pot[0][c]; a(i,j,k,c)=pot[1][c]; }
    });
    for (int pass=0;pass<passes;++pass) {
        Box nextbox=grow(box,passes-pass);
        FArrayBox next(nextbox,3); auto out=next.array(); auto old=pulse.const_array();
        LoopOnCpu(nextbox,[&](int i,int j,int k) {
            for (int c=0;c<3;++c) {
                // Match the original per-axis seven-point Laplacian and its
                // bounded deconvolution update. No coarse/fine interpolation
                // is used to prescribe this external analytic/stencil source.
                Real lap=0;
                lap+=(old(i+1,j,k,c)-2*old(i,j,k,c)+old(i-1,j,k,c))/(dx[0]*dx[0]);
                lap+=(old(i,j+1,k,c)-2*old(i,j,k,c)+old(i,j-1,k,c))/(dx[0]*dx[0]);
                lap+=(old(i,j,k+1,c)-2*old(i,j,k,c)+old(i,j,k-1,c))/(dx[0]*dx[0]);
                out(i,j,k,c)=old(i,j,k,c)-p.correction_strength*dx[0]*dx[0]*lap;
            }
        });
        pulse=std::move(next);
    }
    auto corrected=pulse.const_array(); auto out=result.array();
    // Curl each component separately, as in the original decomposition.
    LoopOnCpu(box,[&](int i,int j,int k) {
        for (int c=0;c<3;++c) {
            int a1=(c+1)%3,a2=(c+2)%3;
            IntVect plus1(i,j,k),minus1(i,j,k),plus2(i,j,k),minus2(i,j,k);
            ++plus1[a1]; --minus1[a1]; ++plus2[a2]; --minus2[a2];
            auto curl=[&](auto const& field) {
                return (field(plus1,a2)-field(minus1,a2))/(2*dx[0])
                    -(field(plus2,a1)-field(minus2,a1))/(2*dx[0]);
            };
            out(i,j,k,c)=curl(b)+curl(corrected);
        }
    });
    return result;
}
inline FArrayBox force(Profile const& p,Box const& box,Geometry const& g,Real t,Real epsilon_tau_ratio=0) {
    FArrayBox result(box,3); result.setVal(0);
    if (t<=p.cutoff_start) return result;
    Real epsilon=std::min({p.epsilon,.2*(p.T-t),.1*phase_limit(p,t)});
    if (epsilon_tau_ratio>0) epsilon=std::min(epsilon,epsilon_tau_ratio*(p.T-t));
    if (!(epsilon>0) || t+epsilon>=p.T || t-epsilon==t || t+epsilon==t)
        Abort("unresolvable paper force derivative window");
    Real before_time=t-epsilon>=0 ? t-epsilon : t;
    Real denominator=t-epsilon>=0 ? 2*epsilon : epsilon;
    auto current=velocity(p,grow(box,1),g,t);
    auto before=velocity(p,box,g,before_time);
    auto after=velocity(p,box,g,t+epsilon);
    auto u=current.const_array(), um=before.const_array(),up=after.const_array(); auto out=result.array();
    Real dx=g.CellSize(0);
    LoopOnCpu(box,[&](int i,int j,int k) {
        for (int c=0;c<3;++c) {
            Real advective=0,lap=0;
            for (int d=0;d<3;++d) {
                IntVect plus(i,j,k),minus(i,j,k); ++plus[d]; --minus[d];
                advective+=u(i,j,k,d)*(u(plus,c)-u(minus,c))/(2*dx);
                lap+=(u(plus,c)-2*u(i,j,k,c)+u(minus,c))/(dx*dx);
            }
            out(i,j,k,c)=(up(i,j,k,c)-um(i,j,k,c))/denominator+advective-p.viscosity*lap;
        }
    });
    return result;
}
}
#endif
