// Correctness check, not a performance benchmark. Compare parallel box forcing
// against the unchanged serial stencil, including cache reuse and eviction.
#include "ns_case.H"

int main(int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        using namespace amrex;
        auto const& opt=ns_case::options();
        if (opt.max_dt<1e99 && (!ns_case::reached_end(.552-2.e-15,.552,1.e-18)
            || ns_case::reached_end(.552-.00025,.552,.00025)))
            Abort("endpoint roundoff check failed");
        for (int i=0;i<opt.plot_times.size();++i) {
            Real t=opt.plot_times[i];
            if (!ns_case::plot_due(t) || !ns_case::plot_due(t+2.e-13) || ns_case::plot_due(t+2.e-12))
                Abort("prescribed output acceptance tolerance mismatch");
            Real dt=i+1<opt.plot_times.size() ? opt.plot_times[i+1]-t : 1.;
            if (ns_case::limit_plot_dt(t,dt)!=dt || ns_case::limit_plot_dt(t,dt/4)!=dt/4)
                Abort("output clock enlarged a bound or skipped an event");
            if (ns_case::limit_plot_dt(t-5.e-13,dt)<dt*.99)
                Abort("accepted output event caused a duplicate tiny step");
        }
        if (opt.force!="shear" && opt.force!="paper") Abort("check requires shear or paper forcing");
        int requested=ns_case::force_thread_limit(), actual=1;
#ifdef AMREX_USE_OMP
#pragma omp parallel num_threads(requested) if(requested > 1)
        {
#pragma omp single
            actual=OpenMP::get_num_threads();
        }
#endif
        if (actual!=requested) Abort("runtime did not supply the requested bounded thread team");
        Real error=0;
        for (int lev=0;lev<2;++lev) {
            int n=16*(lev+1);
            Box full(IntVect(0),IntVect(n-1));
            BoxArray boxes(full); boxes.maxSize(8);
            DistributionMapping dm(boxes);
            RealBox domain({-1.,-1.,-1.},{1.,1.,1.});
            int periodic[3]={1,1,1}; Geometry g(full,&domain,0,periodic);
            auto dx=g.CellSizeArray(); auto lo=g.ProbLoArray();
            Real nu=opt.force=="paper" ? ns_case::profile().viscosity : 0.01;
            MultiFab values(boxes,dm,3,0);
            for (Real time : {0.,0.55,0.85,0.985}) {
                Real end=time+0.0001;
                values.setVal(0.125);
                ns_case::add_force(lev,values,g,time,end,nu);
                for (MFIter mfi(values);mfi.isValid();++mfi) {
                    FArrayBox f0,f1;
                    if (opt.force=="paper") {
                        f0=ns_paper::force(ns_case::profile(),mfi.validbox(),g,time,opt.epsilon_tau_ratio);
                        f1=ns_paper::force(ns_case::profile(),mfi.validbox(),g,end,opt.epsilon_tau_ratio);
                    }
                    auto a=values.const_array(mfi),b=f0.const_array(),c=f1.const_array();
                    LoopOnCpu(mfi.validbox(),[&](int i,int j,int k) {
                        Real x=lo[0]+(i+.5)*dx[0],y=lo[1]+(j+.5)*dx[1],z=lo[2]+(k+.5)*dx[2];
                        for (int d=0;d<3;++d) {
                            Real first=opt.force=="paper" ? b(i,j,k,d) : ns_case::force_value(opt.force,x,y,z,time,d,nu);
                            Real second=opt.force=="paper" ? c(i,j,k,d) : ns_case::force_value(opt.force,x,y,z,end,d,nu);
                            Real expected=.125+.5*(first+second);
                            if (!std::isfinite(a(i,j,k,d))) Abort("non-finite threaded forcing");
                            error=std::max(error,std::abs(a(i,j,k,d)-expected));
                        }
                    });
                }
                if (opt.force=="paper") {
                    auto kept=ns_case::paper_force(lev,values,g,time);
                    if (kept!=ns_case::paper_force(lev,values,g,time)) Abort("force cache miss for identical request");
                    Real norm=kept->norm0(0);
                    auto later=ns_case::paper_force(lev,values,g,end);
                    auto latest=ns_case::paper_force(lev,values,g,end+.0001);
                    if (kept->norm0(0)!=norm) Abort("eviction invalidated a retained force field");
                    if (kept==latest || kept==later) Abort("different force times share storage");
                }
            }
        }
        if (error!=0) Abort("threaded source differs from serial per-box evaluation");
        Print()<<"NS_THREADING_RESULT {\"actual_threads\":"<<actual
            <<",\"force_linf_difference\":"<<error<<",\"passed\":true}\n";
    }
    amrex::Finalize();
}
