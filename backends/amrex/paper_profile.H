#ifndef NS_PAPER_PROFILE_H
#define NS_PAPER_PROFILE_H

// Pointwise port of profile.py's finite, cubic Appendix-B surrogate.
// Cubic coefficients are exported from the exact recorded Python interpolants.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ns_paper {
using Vec = std::array<double,3>;
constexpr double pi=3.1415926535897932384626433832795;
inline double number(std::istream& in) {
    // iostream extraction can reject valid subnormal spline coefficients.
    std::string token; in>>token; char* end=nullptr;
    double value=std::strtod(token.c_str(),&end);
    if (!in || end!=token.c_str()+token.size() || !std::isfinite(value))
        throw std::runtime_error("invalid finite table number");
    return value;
}
struct Spline {
    std::vector<double> x;
    std::vector<std::array<double,4>> c;
    double inv_dx;
    explicit Spline(std::istream& in) {
        int n=0; in>>n;
        if (n<2 || n>131073) throw std::runtime_error("invalid table length");
        x.resize(n); c.resize(n-1);
        for (auto& v:x) v=number(in);
        for (int i=1;i<n;++i) if (x[i]<=x[i-1]) throw std::runtime_error("unordered knots");
        for (auto& row:c) for (auto& v:row) v=number(in);
        inv_dx=1/(x[1]-x[0]);
    }
    double operator()(double q) const {
        q=std::clamp(q,x.front(),x.back());
        int i=std::clamp(int((q-x.front())*inv_dx),0,int(c.size())-1);
        // Handle floating-point placement at a tabulation knot exactly as PPoly.
        while (i+1<int(c.size()) && q>=x[i+1]) ++i;
        while (i>0 && q<x[i]) --i;
        double s=q-x[i]; auto const& a=c[i];
        return ((a[0]*s+a[1])*s+a[2])*s+a[3];
    }
};
inline double bump(double s) { return std::abs(s)<1 ? std::exp(1-1/(1-s*s)) : 0; }
inline double step(double s) {
    if (s<=0) return 0;
    if (s>=1) return 1;
    double a=std::exp(-1/s), b=std::exp(-1/(1-s)); return a/(a+b);
}
inline double plateau(double s,double inner,double outer) { return 1-step((std::abs(s)-inner)/(outer-inner)); }

struct Profile {
    std::map<std::string,double> parameters;
    std::map<std::string,Spline> tables;
    double h,T,R,H,U,swirl,domain,xa,xb,eta_support,eta_taper,axial_slope,axis_offset,swirl_bias;
    double cutoff_start,cutoff_end,loc_inner,loc_outer,pulse_rtheta,pulse_rz,radial_freq,axial_freq,log_freq;
    double time_width,scale_ratio,correction_strength,epsilon,phase_step,phase_rate,viscosity;
    int mode,levels,mode_stride,correction_passes;
    bool pulses,fully_localized;
    double get(std::string const& key) const { return parameters.at(key); }
    explicit Profile(std::string const& path) {
        std::ifstream in(path); std::string magic; int n;
        in>>magic>>n;
        if (!in || magic!="NS_PAPER_TABLES_V1" || n<1 || n>100) throw std::runtime_error("invalid profile bundle");
        for (int i=0;i<n;++i) {
            std::string key; double value; in>>key>>value;
            if (!in || !std::isfinite(value) || !parameters.emplace(key,value).second)
                throw std::runtime_error("invalid or duplicate profile parameter");
        }
        in>>n; if (!in || n!=4) throw std::runtime_error("expected four profile tables");
        for (int i=0;i<n;++i) {
            std::string name; in>>name;
            if (!tables.emplace(name,Spline(in)).second) throw std::runtime_error("duplicate profile table");
        }
        for (auto name:{"radial_u","radial_phi","axis_phi","exterior"})
            if (!tables.contains(name)) throw std::runtime_error("missing named profile table");
        std::string extra; if (in>>extra) throw std::runtime_error("trailing profile data");
        h=get("h"); T=get("t_star"); R=get("base_radius"); H=get("base_height"); U=get("velocity_scale");
        swirl=get("swirl_ratio"); domain=get("half_domain"); xa=get("paper_annulus_xa"); xb=get("paper_annulus_xb");
        eta_support=get("paper_eta_support"); eta_taper=get("paper_eta_taper"); axial_slope=get("paper_axial_slope");
        axis_offset=get("paper_axis_offset"); swirl_bias=get("paper_swirl_bias");
        cutoff_start=get("paper_time_cutoff_start"); cutoff_end=get("paper_time_cutoff_end");
        loc_inner=get("localization_inner"); loc_outer=get("localization_outer");
        pulses=get("pulses_enabled")!=0; pulse_rtheta=get("pulse_rtheta_strength"); pulse_rz=get("pulse_rz_strength");
        mode=int(get("pulse_azimuthal_mode")); radial_freq=get("pulse_radial_frequency");
        axial_freq=get("pulse_axial_frequency"); log_freq=get("pulse_log_frequency"); time_width=get("pulse_time_width");
        levels=int(get("pulse_hierarchy_levels")); mode_stride=int(get("pulse_mode_stride")); scale_ratio=get("pulse_scale_ratio");
        correction_strength=get("pulse_correction_strength"); correction_passes=int(get("pulse_correction_passes"));
        epsilon=get("derivative_epsilon"); phase_step=get("forcing_phase_step"); phase_rate=get("forcing_log_rate_bound");
        viscosity=get("viscosity");
        fully_localized=parameters.contains("fully_localized") && get("fully_localized")!=0;
        if (!(0<h && h<.01 && R>0 && H>0 && domain>loc_outer && loc_outer>loc_inner && loc_inner>0
              && xb>xa && xa>0 && eta_taper>eta_support && eta_taper<1 && cutoff_end>cutoff_start
              && T>cutoff_end && mode>0 && levels>0 && levels<=16 && mode_stride>=0
              && correction_passes>=0 && correction_passes<=4 && epsilon>0 && phase_step>0 && phase_rate>=0))
            throw std::runtime_error("profile parameters outside the supported finite case");
    }
    double activation(double t) const { return step((t-cutoff_start)/(cutoff_end-cutoff_start)); }
    double wrap(double x) const {
        if (x>=-domain && x<domain) return x;
        return x-2*domain*std::floor((x+domain)/(2*domain));
    }
    // Return background and primary pulse potentials. Correction is a stencil.
    std::array<Vec,2> potentials(double x,double y,double z,double t,double dx) const {
        if (!(t<T) || !std::isfinite(t) || !(dx>0)) throw std::runtime_error("invalid profile point time or spacing");
        double ramp=activation(t); if (ramp==0) return {};
        x=wrap(x); y=wrap(y); z=wrap(z);
        double tau=T-t, radius=std::hypot(x,y), theta=std::atan2(y,x), zeta=z/H, D=.5-h;
        double q=tau+std::pow(std::abs(zeta),1/D), z2=zeta*zeta;
        for (int i=0;i<7;++i) {
            double power=std::pow(q,2*h), residual=q-z2*power-tau, derivative=1-2*h*z2*power/q;
            q=std::max(tau,q-residual/derivative);
        }
        double eta=zeta/std::pow(q,D), X=(radius/R)*(radius/R)/(2*q);
        double cosine=radius>0?x/radius:0, sine=radius>0?y/radius:0;
        double integral=tables.at("radial_u")(X);
        double phi=tables.at("radial_phi")(X)-tables.at("radial_phi")(tables.at("radial_phi").x.back());
        double window=plateau(eta,eta_support,eta_taper);
        double axial=(axial_slope*eta+axis_offset)*window;
        double axis_phi=tables.at("axis_phi")(eta)*(1+swirl_bias*eta)*window;
        double a_theta=U*ramp*R*R*std::pow(q,1-(.5+h))*integral*axial/std::max(radius,.25*dx);
        double a_z=-swirl*U*ramp*R*std::pow(q,.5-(.5+h))*phi*axis_phi;
        double axial_loc=plateau(z,loc_inner,loc_outer);
        double localization=plateau(radius,loc_inner,loc_outer)*axial_loc;
        if (fully_localized) a_z*=localization;
        double blend=step((X-xa)/(xb-xa));
        double exterior=swirl*U*ramp*std::exp(-1/.55)*std::pow(xa,1+h)*std::pow(2.,1+h)*std::pow(R,1+2*h);
        a_z+=exterior*blend*tables.at("exterior")(radius)*axial_loc;
        a_theta*=localization;
        Vec background{-sine*a_theta,cosine*a_theta,a_z};
        double pulse_theta=0,pulse_z=0;
        if (pulses && (pulse_rtheta>0 || pulse_rz>0)) {
            double envelope=bump((2*X-xa-xb)/(xb-xa))*bump(eta/eta_support)*localization;
            if (envelope!=0) {
                double log_slot=-std::log2(tau), log_phase=2*pi*log_freq*log_slot;
                double local_v=U*ramp*std::pow(q,-.5-h+.5*h), local_length=R*std::sqrt(q);
                double radial_phase=radial_freq*std::log(std::max(X,.25*xa)/xa);
                auto gate=[&](double offset) {
                    double s=log_slot-offset+.5;
                    double distance=s-std::floor(s)-.5;
                    return bump(distance/time_width);
                };
                for (int l=0;l<levels;++l) {
                    double amplitude=std::pow(scale_ratio,l), freq=1+.65*l, phase=.41*l*pi;
                    int m=mode+l*mode_stride;
                    double one=m*theta+freq*radial_phase+(1+.17*l)*log_phase+phase;
                    double two=(m+1)*theta+freq*radial_phase+freq*axial_freq*eta-(.73+.09*l)*log_phase+pi/3-phase;
                    pulse_z+=pulse_rtheta*amplitude*gate(std::fmod(.18+.21*l,1.))*local_v*local_length*envelope*std::sin(one)/m;
                    pulse_theta+=pulse_rz*amplitude*gate(std::fmod(.68+.21*l,1.))*local_v*local_length*envelope*std::cos(two)/(m+1);
                }
            }
        }
        return {background,Vec{-sine*pulse_theta,cosine*pulse_theta,pulse_z}};
    }
};
}
#endif
