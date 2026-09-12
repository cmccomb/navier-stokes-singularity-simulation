#include <AMReX_PlotFileDataImpl.H>
#include "ns_case.H"

int main(int argc, char** argv) {
    amrex::Initialize(argc,argv);
    {
        using namespace amrex;
        ParmParse pp; std::string path; Real nu=0.01;
        pp.get("plot",path); pp.query("nu",nu);
        bool report_difference=false; pp.query("report_difference",report_difference);
        PlotFileDataImpl data(path);
        std::string compare;
        std::unique_ptr<PlotFileDataImpl> reference;
        if (pp.query("compare",compare)) {
            reference=std::make_unique<PlotFileDataImpl>(compare);
            Real time_tolerance=report_difference ? 1e-12 : 0;
            if (std::abs(reference->time()-data.time())>time_tolerance || reference->finestLevel()!=data.finestLevel()
                || reference->varNames()!=data.varNames() || reference->probLo()!=data.probLo() || reference->probHi()!=data.probHi())
                Abort("restart comparison has different time or levels");
        }
        if (report_difference && !reference) Abort("difference report requires a reference plot");
        Vector<std::string> expected{"velx","vely","velz","forcing_x","forcing_y","forcing_z"};
        if (data.varNames()!=expected || data.spaceDim()!=3)
            Abort("native archive is missing velocity/force components");
        Real max_force_error=0, max_speed=0, max_velocity_difference=0; Long stored=0;
        Real difference_sq=0, reference_sq=0,volume=0;
        int reference_ratio=1;
        for (int lev=0; lev<=data.finestLevel(); ++lev) {
            MultiFab mf=data.get(lev);
            MultiFab comparison;
            if (reference) {
                auto source=reference->get(lev);
                if (!report_difference && (source.boxArray()!=mf.boxArray() || reference->probDomain(lev)!=data.probDomain(lev)))
                    Abort("restart mesh changed");
                IntVect coarse_length=data.probDomain(lev).length();
                IntVect fine_length=reference->probDomain(lev).length();
                int ratio=fine_length[0]/coarse_length[0];
                if (ratio<1 || fine_length!=ratio*coarse_length ||
                    coarsen(reference->probDomain(lev),ratio)!=data.probDomain(lev))
                    Abort("comparison requires an aligned integer finer reference");
                if (lev==0) reference_ratio=ratio;
                else if (reference_ratio!=ratio) Abort("reference refinement ratio changes between levels");
                BoxArray coarse_boxes=source.boxArray(); coarse_boxes.coarsen(ratio);
                if (coarse_boxes.numPts()!=mf.boxArray().numPts())
                    Abort("comparison refinement regions have different physical coverage");
                comparison.define(mf.boxArray(),mf.DistributionMap(),mf.nComp(),0);
                comparison.setVal(std::numeric_limits<Real>::quiet_NaN());
                if (ratio==1) comparison.ParallelCopy(source);
                else {
                    MultiFab restricted(coarse_boxes,source.DistributionMap(),source.nComp(),0);
                    average_down(source,restricted,0,source.nComp(),IntVect(ratio));
                    comparison.ParallelCopy(restricted);
                }
            }
            auto dx=data.cellSize(lev); auto lo=data.probLo();
            iMultiFab mask;
            if (lev<data.finestLevel()) mask=makeFineMask(mf.boxArray(),mf.DistributionMap(),data.boxArray(lev+1),data.refRatioVect(lev),1,0);
            else { mask.define(mf.boxArray(),mf.DistributionMap(),1,0); mask.setVal(1); }
            Long active=0; Real dv=dx[0]*dx[1]*dx[2];
            auto hi=data.probHi();
            RealBox physical(lo.data(),hi.data()); int periodic[3]={1,1,1};
            Geometry geometry(data.probDomain(lev),&physical,data.coordSys(),periodic);
            stored+=mf.boxArray().numPts();
            for (MFIter mfi(mf);mfi.isValid();++mfi) {
                auto a=mf.const_array(mfi); auto m=mask.const_array(mfi);
                auto b=reference ? comparison.const_array(mfi) : a;
                FArrayBox paper_force;
                if (ns_case::options().force=="paper")
                    paper_force=ns_paper::force(ns_case::profile(),mfi.validbox(),geometry,data.time(),ns_case::options().epsilon_tau_ratio);
                auto external=paper_force.const_array();
                LoopOnCpu(mfi.validbox(),[&](int i,int j,int k) {
                    Real x=lo[0]+(i+0.5)*dx[0], y=lo[1]+(j+0.5)*dx[1], z=lo[2]+(k+0.5)*dx[2];
                    Real speed2=0;
                    if (m(i,j,k)) ++active;
                    for (int c=0;c<3;++c) {
                        if (!std::isfinite(a(i,j,k,c)) || !std::isfinite(a(i,j,k,c+3)) || !std::isfinite(b(i,j,k,c)))
                            Abort("non-finite archived vector");
                        speed2+=a(i,j,k,c)*a(i,j,k,c);
                        max_velocity_difference=std::max(max_velocity_difference,std::abs(a(i,j,k,c)-b(i,j,k,c)));
                        if (m(i,j,k)) {
                            Real difference=a(i,j,k,c)-b(i,j,k,c);
                            difference_sq+=difference*difference*dv;
                            reference_sq+=b(i,j,k,c)*b(i,j,k,c)*dv;
                        }
                        Real f=ns_case::options().force=="paper" ? external(i,j,k,c)
                            : ns_case::force_value(ns_case::options().force,x,y,z,data.time(),c,nu);
                        max_force_error=std::max(max_force_error,std::abs(a(i,j,k,c+3)-f));
                    }
                    max_speed=std::max(max_speed,std::sqrt(speed2));
                });
            }
            volume+=active*dv;
        }
        if (max_force_error>1e-12) Abort("saved forcing differs from instantaneous external body force");
        if (!report_difference && max_velocity_difference>1e-12) Abort("restarted velocity differs from uninterrupted trajectory");
        Print()<<std::setprecision(17)<<"NS_ARCHIVE_RESULT {\"time\":"<<data.time()
            <<",\"step\":"<<data.levelStep(0)<<",\"levels\":"<<data.finestLevel()+1
            <<",\"stored_cells\":"<<stored<<",\"force_linf_error\":"<<max_force_error
            <<",\"peak_speed\":"<<max_speed<<",\"compared\":"<<(reference ? "true" : "false")
            <<",\"report_difference\":"<<(report_difference ? "true" : "false")
            <<",\"reference_resolution_ratio\":"<<reference_ratio
            <<",\"time_difference\":"<<(reference ? data.time()-reference->time() : 0)
            <<",\"velocity_linf_difference\":"<<max_velocity_difference
            <<",\"velocity_l2_difference\":"<<std::sqrt(difference_sq/volume)
            <<",\"reference_velocity_l2\":"<<std::sqrt(reference_sq/volume)<<",\"composite_volume\":"<<volume<<"}\n";
    }
    amrex::Finalize();
}
