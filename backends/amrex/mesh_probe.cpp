// Generate real AMReX boxes using the production tagger; allocate no flow fields.
#include "ns_case.H"
#include <AMReX_AmrMesh.H>

class MeshProbe : public amrex::AmrMesh {
public:
    void build() { MakeNewGrids(0.0); }
    void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real, int) override {
        ns_case::tag_fixed_region(lev, tags, geom[lev]);
    }
    void report() const {
        using namespace amrex;
        Long stored=0, active=0;
        Print()<<std::setprecision(17)<<"NS_MESH_PROBE {\"schema_version\":1,\"levels\":[";
        for(int lev=0;lev<=finest_level;++lev) {
            if(lev) Print()<<",";
            Long cells=grids[lev].numPts(); stored+=cells; active+=cells;
            if(lev<finest_level) active-=grids[lev+1].numPts()/8;
            Print()<<"{\"level\":"<<lev<<",\"cells\":"<<cells
                   <<",\"box_count\":"<<grids[lev].size()<<",\"dx\":"<<geom[lev].CellSize(0)<<",\"boxes\":[";
            for(int k=0;k<grids[lev].size();++k) {
                if(k) Print()<<",";
                auto b=grids[lev][k]; auto lo=b.smallEnd(), sz=b.length();
                Print()<<"{\"index_lo\":["<<lo[0]<<","<<lo[1]<<","<<lo[2]
                       <<"],\"shape\":["<<sz[0]<<","<<sz[1]<<","<<sz[2]<<"]}";
            }
            Print()<<"]}";
        }
        struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
        double peak=usage.ru_maxrss;
#ifdef __APPLE__
        peak/=1024*1024;
#else
        peak/=1024;
#endif
        Print()<<"],\"stored_cells\":"<<stored<<",\"active_cells\":"<<active<<",\"peak_rss_mib\":"<<peak<<"}\n";
    }
};

int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    { MeshProbe mesh; mesh.build(); mesh.report(); }
    amrex::Finalize();
}
