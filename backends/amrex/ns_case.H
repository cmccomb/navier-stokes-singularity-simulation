#ifndef NS_CASE_H
#define NS_CASE_H

// CPU, constant-density adapter for the coupled incflo integrator. The finite
// paper surrogate is a prescribed source, not the infinite proof construction.
#include "paper_fields.H"
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_TagBox.H>
#include <AMReX_OpenMP.H>
#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <memory>
#include <sys/resource.h>
#ifndef NS_ADAPTER_SHA256
#define NS_ADAPTER_SHA256 "not-built-as-integrator"
#endif

namespace ns_case {
using namespace amrex;
constexpr Real pi = 3.1415926535897932384626433832795;
struct Options {
    std::string force = "none";
    Real max_dt = 1e100;
    Real quiescent_max_dt = 0;
    Real epsilon_tau_ratio = 0;
    std::string table_file;
    Vector<Real> widths;
    Vector<Real> plot_times;
    bool diagnostics = false;
    int force_threads = 1;
    Options() {
        ParmParse pp("ns");
        pp.query("force", force);
        pp.query("max_dt", max_dt);
        pp.query("quiescent_max_dt", quiescent_max_dt);
        pp.query("epsilon_tau_ratio", epsilon_tau_ratio);
        pp.query("table_file", table_file);
        pp.queryarr("refine_half_width", widths);
        pp.queryarr("plot_times", plot_times);
        pp.query("diagnostics", diagnostics);
        pp.query("force_threads", force_threads);
        int expected_threads=0;
        pp.query("expected_omp_threads", expected_threads);
        if (expected_threads < 0 || (expected_threads && expected_threads!=OpenMP::get_max_threads()))
            Abort("OpenMP thread configuration does not match ns.expected_omp_threads");
        if (force_threads < 1) Abort("ns.force_threads must be positive");
#ifndef AMREX_USE_OMP
        if (force_threads != 1) Abort("ns.force_threads > 1 requires NS_ENABLE_OPENMP=ON");
#endif
        if (force != "none" && force != "rest" && force != "oscillation" && force != "shear" && force != "paper")
            Abort("ns.force must be none, rest, oscillation, shear, or paper");
        if (!std::isfinite(max_dt) || max_dt <= 0) Abort("ns.max_dt must be finite and positive");
        if (!std::isfinite(quiescent_max_dt) || quiescent_max_dt < 0 ||
            !std::isfinite(epsilon_tau_ratio) || epsilon_tau_ratio < 0)
            Abort("quiescent dt and derivative ratio must be finite and nonnegative");
        if (force == "paper" && table_file.empty()) Abort("paper force requires ns.table_file");
        for (int l=0; l<widths.size(); ++l)
            if (!std::isfinite(widths[l]) || widths[l] <= 0 || widths[l] >= (l ? widths[l-1] : 1.0))
                Abort("fixed refinement widths must decrease inside [-1,1]^3");
        for (int i=0;i<plot_times.size();++i)
            if (!std::isfinite(plot_times[i]) || plot_times[i]<0 ||
                (i && plot_times[i]-plot_times[i-1]<=1.e-11))
                Abort("prescribed output times must be finite and strictly separated");
    }
};
inline Options const& options() { static Options o; return o; }
inline Real limit_plot_dt(Real time, Real dt) {
    auto const& events=options().plot_times;
    auto next=std::upper_bound(events.begin(),events.end(),time+1.e-12);
    return next==events.end() ? dt : std::min(dt,*next-time);
}
inline bool plot_due(Real time) {
    auto const& events=options().plot_times;
    auto next=std::lower_bound(events.begin(),events.end(),time-1.e-12);
    return next!=events.end() && std::abs(*next-time)<=1.e-12;
}
inline bool reached_end(Real time, Real end, Real dt) {
    // Our diagnostic/archive contract accepts 2e-14 endpoint roundoff. Do not
    // take a new tiny step after accepting that endpoint as an output event.
    Real tolerance=options().max_dt<1e99 ? Real(2.e-14) : Real(1.e-12)*dt;
    return time>=end-tolerance;
}
inline ns_paper::Profile const& profile() {
    static ns_paper::Profile p(options().table_file); return p;
}
inline int force_thread_limit() {
    return std::min(options().force_threads, OpenMP::get_max_threads());
}
inline void validate(Real fixed_dt, int regrid_int, bool constant_density, Real density, Real nu) {
    auto const& o=options();
    if (fixed_dt > 0 && (o.force != "none" || o.max_dt < 1e99))
        Abort("use ns.max_dt, not the upstream fixed_dt CFL override");
    if (!o.widths.empty() && regrid_int > 0)
        Abort("pre-refined tests require amr.regrid_int <= 0");
    if (o.force != "none" && (!constant_density || density != 1.0))
        Abort("the acceleration adapter requires constant unit density");
    if (o.force == "paper" && nu != profile().viscosity)
        Abort("incflo.mu must match the exported paper viscosity");
}
inline Real limit_dt(Real time, Real dt, bool zero_cfl=false) {
    auto const& o=options();
    if (o.force == "paper") {
        auto const& p=profile();
        bool before_activation=p.cutoff_start-time>Real(16)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(p.cutoff_start));
        // Only skip over an exactly quiescent interval, ending at activation.
        // A nonzero velocity/force CFL must retain the upstream stability bound.
        if (before_activation && zero_cfl && o.quiescent_max_dt>0)
            return std::min(o.quiescent_max_dt,p.cutoff_start-time);
        dt=std::min(dt,ns_paper::phase_limit(p,time));
        if (before_activation) dt=std::min(dt,p.cutoff_start-time);
    }
    return std::min(dt,o.max_dt);
}

struct ForceRecord {
    Real time, dx;
    std::shared_ptr<MultiFab> values;
};
inline std::shared_ptr<MultiFab const> paper_force(int level, MultiFab const& layout, Geometry const& g, Real time) {
    // The cache is owned by the integrator thread. Only independent boxes below
    // run concurrently; never mutate the cache or initialize profile data there.
    if (OpenMP::in_parallel()) Abort("paper_force must be called outside an OpenMP region");
    auto const& p=profile();
    for (int c=0;c<3;++c)
        if (g.ProbLo(c)!=-p.domain || g.ProbHi(c)!=p.domain || !g.isPeriodic(c))
            Abort("paper source requires its full periodic domain");
    static Vector<std::deque<ForceRecord>> cache;
    static bool registered=[] { ExecOnFinalize([] { cache.clear(); }); return true; }();
    (void)registered;
    if (cache.size()<=level) cache.resize(level+1);
    auto& records=cache[level];
    for (auto const& record:records)
        if (record.time==time && record.dx==g.CellSize(0) &&
            record.values->boxArray()==layout.boxArray() &&
            record.values->DistributionMap()==layout.DistributionMap()) return record.values;
    auto value=std::make_shared<MultiFab>(layout.boxArray(),layout.DistributionMap(),3,0);
    [[maybe_unused]] int threads=force_thread_limit();
    Real epsilon_tau_ratio=options().epsilon_tau_ratio;
#ifdef AMREX_USE_OMP
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
    {
        // Whole boxes bound scratch memory to one stencil working set per worker.
        // Tiling would repeat padded potential evaluations around every tile.
        for (MFIter mfi(*value);mfi.isValid();++mfi) {
            auto source=ns_paper::force(p,mfi.validbox(),g,time,epsilon_tau_ratio);
            (*value)[mfi].copy(source);
        }
    }
    records.push_back({time,g.CellSize(0),value});
    // f(t_n) and f(t_n+dt) suffice for MOL. Shared ownership also makes a
    // midpoint/Godunov request safe if it evicts a still-used endpoint.
    while (records.size()>2) records.pop_front();
    return value;
}

inline Real force_value(std::string const& mode, Real x, Real y, Real z, Real t, int c, Real nu) {
    if (mode == "oscillation") return 0.1*(c+1)*5.0*std::sin(10*t);
    if (mode != "shear") return 0;
    Real coord[3]={y,z,x};
    Real advector[3]={z,x,y};
    return (1+nu*pi*pi*t)*std::sin(pi*coord[c])
        + t*t*pi*std::sin(pi*advector[c])*std::cos(pi*coord[c]);
}
inline Real exact_value(int problem, Real x, Real y, Real z, Real t, int c, Real nu) {
    auto const& mode=options().force;
    if (mode == "rest") return 0;
    if (mode == "oscillation") return 0.1*(c+1)*0.5*(1-std::cos(10*t));
    if (mode == "shear") { Real coord[3]={y,z,x}; return t*std::sin(pi*coord[c]); }
    if (problem == 2) {
        if (c==0) return 1-std::cos(pi*(x-t))*std::sin(pi*(y-t))*std::exp(-2*pi*pi*nu*t);
        if (c==1) return 1+std::sin(pi*(x-t))*std::cos(pi*(y-t))*std::exp(-2*pi*pi*nu*t);
        return 0;
    }
    Abort("diagnostic exact reference is unavailable for this case");
    return 0;
}
inline void add_force(int level, MultiFab& f, Geometry const& g, Real time, Real end, Real nu) {
    if (OpenMP::in_parallel()) Abort("add_force must be called outside an OpenMP region");
    auto const& mode=options().force;
    if (mode == "none" || mode == "rest") return;
    [[maybe_unused]] int threads=force_thread_limit();
    if (mode == "paper") {
        if (time<=profile().cutoff_start && (end<0 || end<=profile().cutoff_start)) return;
        auto first=paper_force(level,f,g,time);
        auto second=end>=0 ? paper_force(level,f,g,end) : first;
#ifdef AMREX_USE_OMP
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
        {
            for (MFIter mfi(f);mfi.isValid();++mfi) {
                auto a=f.array(mfi); auto b=first->const_array(mfi),c=second->const_array(mfi);
                LoopOnCpu(mfi.validbox(),[&](int i,int j,int k) {
                    for (int d=0;d<3;++d) a(i,j,k,d)+=.5*(b(i,j,k,d)+c(i,j,k,d));
                });
            }
        }
        return;
    }
    auto dx=g.CellSizeArray(); auto lo=g.ProbLoArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
    {
        for (MFIter mfi(f); mfi.isValid(); ++mfi) {
            auto a=f.array(mfi);
            LoopOnCpu(mfi.validbox(), [&](int i,int j,int k) {
                Real x=lo[0]+(i+0.5)*dx[0], y=lo[1]+(j+0.5)*dx[1], z=lo[2]+(k+0.5)*dx[2];
                for (int c=0;c<3;++c) {
                    Real value=force_value(mode,x,y,z,time,c,nu);
                    if (end >= 0) value=0.5*(value+force_value(mode,x,y,z,end,c,nu));
                    a(i,j,k,c)+=value;
                }
            });
        }
    }
}
inline void tag_fixed_region(int level, TagBoxArray& tags, Geometry const& g) {
    auto const& widths=options().widths;
    if (level >= widths.size()) return;
    auto dx=g.CellSizeArray(); auto lo=g.ProbLoArray();
    Real width=widths[level];
    for (int c=0;c<3;++c)
        if (g.ProbLo(c)!=-1 || g.ProbHi(c)!=1 || !g.isPeriodic(c))
            Abort("fixed refinement requires full periodic [-1,1]^3");
    for (MFIter mfi(tags); mfi.isValid(); ++mfi) {
        auto a=tags.array(mfi);
        LoopOnCpu(mfi.validbox(), [&](int i,int j,int k) {
            if (std::abs(lo[0]+(i+0.5)*dx[0]) < width &&
                std::abs(lo[1]+(j+0.5)*dx[1]) < width &&
                std::abs(lo[2]+(k+0.5)*dx[2]) < width) a(i,j,k)=TagBox::SET;
        });
    }
}
inline void diagnose(Vector<MultiFab const*> const& vel, Vector<Geometry> const& geom,
                     Vector<IntVect> const& ratio, int problem, Real time, Real dt, int step, Real nu) {
    if (!options().diagnostics) return;
    Real volume=0, sumsq=0, maxerr=0, peak=0, energy=0;
    Real integral[3]={0,0,0};
    Long stored=0,active=0;
    for (int lev=0;lev<vel.size();++lev) {
        auto const& v=*vel[lev];
        iMultiFab mask;
        if (lev+1 < vel.size()) mask=makeFineMask(v.boxArray(),v.DistributionMap(),vel[lev+1]->boxArray(),ratio[lev],1,0);
        else { mask.define(v.boxArray(),v.DistributionMap(),1,0); mask.setVal(1); }
        auto dx=geom[lev].CellSizeArray(); auto lo=geom[lev].ProbLoArray();
        Real dv=dx[0]*dx[1]*dx[2];
        stored+=v.boxArray().numPts();
        Long active_before=active;
        for (MFIter mfi(v);mfi.isValid();++mfi) {
            auto a=v.const_array(mfi); auto m=mask.const_array(mfi);
            FArrayBox paper_reference;
            if (options().force=="paper") paper_reference=ns_paper::velocity(profile(),mfi.validbox(),geom[lev],time);
            auto target=paper_reference.const_array();
            LoopOnCpu(mfi.validbox(),[&](int i,int j,int k) {
                if (!m(i,j,k)) return;
                ++active; Real speed2=0;
                Real x=lo[0]+(i+0.5)*dx[0], y=lo[1]+(j+0.5)*dx[1], z=lo[2]+(k+0.5)*dx[2];
                for (int c=0;c<3;++c) {
                    Real u=a(i,j,k,c);
                    if (!std::isfinite(u)) Abort("non-finite coupled solution");
                    Real e=u-(options().force=="paper" ? target(i,j,k,c) : exact_value(problem,x,y,z,time,c,nu));
                    sumsq+=e*e*dv; maxerr=std::max(maxerr,std::abs(e));
                    speed2+=u*u; integral[c]+=u*dv;
                }
                peak=std::max(peak,std::sqrt(speed2)); energy+=0.5*speed2*dv;
            });
        }
        // Integer counts avoid O(cell count) roundoff for non-binary spacings.
        volume+=Real(active-active_before)*dv;
    }
    Real domain_volume=1;
    for (int c=0;c<3;++c) domain_volume*=geom[0].ProbHi(c)-geom[0].ProbLo(c);
    if (std::abs(volume-domain_volume)>1e-10) Abort("composite diagnostic volume does not cover the domain");
    // Diagnostics deliberately retain serial summation order. No MPI reductions.
    if (ParallelDescriptor::NProcs()!=1) Abort("diagnostic reductions require a single MPI process");
    struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    Real rss=usage.ru_maxrss;
#ifdef __APPLE__
    rss/=1024*1024;
#else
    rss/=1024;
#endif
    Print()<<std::setprecision(17)<<"NS_INCFLO_RESULT {\"schema_version\":1,\"adapter_sha256\":\""
        <<NS_ADAPTER_SHA256<<"\",\"case\":\""
        <<options().force<<"\",\"problem\":"<<problem<<",\"step\":"<<step<<",\"time\":"<<time
        <<",\"dt\":"<<dt<<",\"levels\":"<<vel.size()<<",\"stored_cells\":"<<stored
        <<",\"active_cells\":"<<active<<",\"volume\":"<<volume<<",\"l2_error\":"<<std::sqrt(sumsq/volume)
        <<",\"openmp_max_threads\":"<<OpenMP::get_max_threads()<<",\"force_threads_limit\":"<<force_thread_limit()
        <<",\"linf_error\":"<<maxerr<<",\"peak_speed\":"<<peak<<",\"energy\":"<<energy
        <<",\"velocity_integral\":["<<integral[0]<<","<<integral[1]<<","<<integral[2]
        <<"],\"peak_rss_mib\":"<<rss<<"}\n";
}
}
#endif
