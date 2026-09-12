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
        if (n<8 || n>256 || n%2) Abort("invalid probe grid");
        static_assert(std::endian::native==std::endian::little && sizeof(Real)==8);
        ns_paper::Profile p(table);
        Box box(IntVect(0),IntVect(n-1));
        RealBox domain({-p.domain,-p.domain,-p.domain},{p.domain,p.domain,p.domain});
        int periodic[3]={1,1,1}; Geometry g(box,&domain,0,periodic);
        auto u=ns_paper::velocity(p,box,g,t),f=ns_paper::force(p,box,g,t);
        auto ua=u.const_array(),fa=f.const_array();
        std::ofstream stream(output,std::ios::binary);
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) for (int k=0;k<n;++k) {
            Real row[6]={ua(i,j,k,0),ua(i,j,k,1),ua(i,j,k,2),fa(i,j,k,0),fa(i,j,k,1),fa(i,j,k,2)};
            stream.write(reinterpret_cast<char const*>(row),sizeof(row));
        }
        stream.close(); if (!stream) Abort("probe output write failed");
        Print()<<std::setprecision(17)<<"NS_FORCE_PROBE {\"n\":"<<n<<",\"time\":"<<t
            <<",\"dx\":"<<g.CellSize(0)<<",\"phase_limit\":"<<ns_paper::phase_limit(p,t)<<"}\n";
    }
    amrex::Finalize();
}
