// Native AMR display sampling: finest available cell, without spatial smoothing.
#include <AMReX_PlotFileDataImpl.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_MFIter.H>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
        using namespace amrex;
        ParmParse pp; std::string plot,output; int n=256;
        pp.get("plot",plot); pp.get("output",output); pp.query("display_n",n);
        if (n<16 || n>2048) Abort("invalid display grid");
        PlotFileDataImpl data(plot);
        Vector<std::string> names{"velx","vely","velz","forcing_x","forcing_y","forcing_z"};
        if (data.spaceDim()!=3 || data.varNames()!=names) Abort("native vectors missing");
        auto lo=data.probLo(),hi=data.probHi();
        for(int d=0;d<3;++d) if(lo[d]!=-1 || hi[d]!=1) Abort("expected full periodic box");
        static_assert(std::endian::native==std::endian::little && sizeof(Real)==8);
        std::vector<Real> pixels(3L*n*n*6,std::numeric_limits<Real>::quiet_NaN());
        int axes[3][3]={{0,2,1},{0,1,2},{1,2,0}}; // horizontal, vertical, fixed
        for(int lev=0;lev<=data.finestLevel();++lev) {
            auto mf=data.get(lev); auto dx=data.cellSize(lev);
            for(MFIter mfi(mf);mfi.isValid();++mfi) {
                auto a=mf.const_array(mfi); auto box=mfi.validbox();
                for(int plane=0;plane<3;++plane) {
                    int h=axes[plane][0],v=axes[plane][1],f=axes[plane][2];
                    int fixed=static_cast<int>(std::floor(-lo[f]/dx[f]));
                    if(fixed<box.smallEnd(f) || fixed>box.bigEnd(f)) continue;
                    for(int i=0;i<n;++i) {
                        int ih=static_cast<int>(std::floor((2.*(i+.5)/n)/dx[h]));
                        if(ih<box.smallEnd(h) || ih>box.bigEnd(h)) continue;
                        for(int j=0;j<n;++j) {
                            int iv=static_cast<int>(std::floor((2.*(j+.5)/n)/dx[v]));
                            if(iv<box.smallEnd(v) || iv>box.bigEnd(v)) continue;
                            IntVect cell; cell[h]=ih; cell[v]=iv; cell[f]=fixed;
                            for(int c=0;c<6;++c) pixels[((plane*n+i)*n+j)*6+c]=a(cell,c);
                        }
                    }
                }
            }
        }
        for(Real value:pixels) if(!std::isfinite(value)) Abort("incomplete/nonfinite slice");
        std::ofstream stream(output,std::ios::binary);
        stream.write(reinterpret_cast<char const*>(pixels.data()),pixels.size()*sizeof(Real));
        stream.close(); if(!stream) Abort("slice export failed");
        Print()<<std::setprecision(17)<<"NS_SLICE_RESULT {\"time\":"<<data.time()
            <<",\"levels\":"<<data.finestLevel()+1<<",\"display_n\":"<<n
            <<",\"planes\":[\"xz\",\"xy\",\"yz\"],\"sampling\":\"finest containing cell; positive-side cell at zero plane; no spatial or temporal smoothing\"}\n";
    }
    amrex::Finalize();
}
