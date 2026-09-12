#include "paper_fields.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <bit>
#include <fstream>
#include <iomanip>
int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
        using namespace amrex;
        ParmParse pp; std::string table,output; int n=16; Real t=.85;
        pp.get("table",table); pp.get("output",output); pp.query("n_cell",n); pp.query("time",t);
        int width=n; pp.query("patch_width",width);
        Vector<int> start(3,(n-width)/2); pp.queryarr("patch_start",start);
        if (n<8 || n>32768 || n%2 || width<1 || width>256 || start.size()!=3)
            Abort("invalid probe grid or bounded patch");
        Real epsilon_tau_ratio=0; pp.query("epsilon_tau_ratio",epsilon_tau_ratio);
        static_assert(std::endian::native==std::endian::little && sizeof(Real)==8);
        ns_paper::Profile p(table);
        Box full(IntVect(0),IntVect(n-1));
        IntVect first(start[0],start[1],start[2]);
        Box box(first,first+IntVect(width-1));
        if (!full.contains(box)) Abort("probe patch is outside the full domain");
        RealBox domain({-p.domain,-p.domain,-p.domain},{p.domain,p.domain,p.domain});
        int periodic[3]={1,1,1}; Geometry g(full,&domain,0,periodic);
        auto u=ns_paper::velocity(p,box,g,t),f=ns_paper::force(p,box,g,t,epsilon_tau_ratio);
        auto ua=u.const_array(),fa=f.const_array();
        std::ofstream stream(output,std::ios::binary);
        for (int i=start[0];i<start[0]+width;++i) for (int j=start[1];j<start[1]+width;++j) for (int k=start[2];k<start[2]+width;++k) {
            Real row[6]={ua(i,j,k,0),ua(i,j,k,1),ua(i,j,k,2),fa(i,j,k,0),fa(i,j,k,1),fa(i,j,k,2)};
            stream.write(reinterpret_cast<char const*>(row),sizeof(row));
        }
        stream.close(); if (!stream) Abort("probe output write failed");
        Print()<<std::setprecision(17)<<"NS_FORCE_PROBE {\"n\":"<<n<<",\"time\":"<<t
            <<",\"patch_width\":"<<width<<",\"dx\":"<<g.CellSize(0)<<",\"phase_limit\":";
        Real limit=ns_paper::phase_limit(p,t);
        if (std::isfinite(limit)) Print()<<std::setprecision(17)<<limit;
        else Print()<<"null";
        Print()<<"}\n";
    }
    amrex::Finalize();
}
