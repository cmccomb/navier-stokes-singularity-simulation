// Native AMR display sampling: finest available cell, without spatial smoothing.
#include <AMReX_PlotFileDataImpl.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Loop.H>
#include <bit>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <tuple>

int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
        using namespace amrex;
        ParmParse pp; std::string plot,output; int n=256; bool peak_xy=false;
        pp.get("plot",plot); pp.get("output",output); pp.query("display_n",n);
        pp.query("peak_xy",peak_xy);
        if (n<16 || n>2048) Abort("invalid display grid");
        PlotFileDataImpl data(plot);
        Vector<std::string> names{"velx","vely","velz","forcing_x","forcing_y","forcing_z"};
        if (data.spaceDim()!=3 || data.varNames()!=names) Abort("native vectors missing");
        auto lo=data.probLo(),hi=data.probHi();
        for(int d=0;d<3;++d) if(lo[d]!=-1 || hi[d]!=1) Abort("expected full periodic box");
        static_assert(std::endian::native==std::endian::little && sizeof(Real)==8);
        Real peak_speed=0; std::array<Real,3> peak_position{0,0,0}; int peak_level=-1;
        if (peak_xy) {
            // Find the actual composite-grid maximum, excluding covered coarse cells.
            for(int lev=0;lev<=data.finestLevel();++lev) {
                auto mf=data.get(lev); auto dx=data.cellSize(lev);
                iMultiFab mask;
                if(lev<data.finestLevel()) mask=makeFineMask(mf.boxArray(),mf.DistributionMap(),
                    data.boxArray(lev+1),data.refRatioVect(lev),1,0);
                else { mask.define(mf.boxArray(),mf.DistributionMap(),1,0); mask.setVal(1); }
                for(MFIter mfi(mf);mfi.isValid();++mfi) {
                    auto a=mf.const_array(mfi); auto m=mask.const_array(mfi);
                    LoopOnCpu(mfi.validbox(),[&](int i,int j,int k) {
                        if(!m(i,j,k)) return;
                        Real speed=std::sqrt(a(i,j,k,0)*a(i,j,k,0)+a(i,j,k,1)*a(i,j,k,1)+a(i,j,k,2)*a(i,j,k,2));
                        if(!std::isfinite(speed)) Abort("nonfinite native velocity");
                        std::array<Real,3> position{lo[0]+(i+.5)*dx[0],lo[1]+(j+.5)*dx[1],lo[2]+(k+.5)*dx[2]};
                        auto tie=[](auto const& p) { return std::tuple(std::abs(p[2]),-p[2],p[1],p[0]); };
                        if(speed>peak_speed || (speed>0 && speed==peak_speed && tie(position)<tie(peak_position))) {
                            peak_speed=speed; peak_position=position; peak_level=lev;
                        }
                    });
                }
            }
        }
        int planes=peak_xy ? 4 : 3;
        std::vector<Real> pixels(static_cast<Long>(planes)*n*n*6,std::numeric_limits<Real>::quiet_NaN());
        int axes[4][3]={{0,2,1},{0,1,2},{1,2,0},{0,1,2}}; // horizontal, vertical, fixed
        for(int lev=0;lev<=data.finestLevel();++lev) {
            auto mf=data.get(lev); auto dx=data.cellSize(lev);
            for(MFIter mfi(mf);mfi.isValid();++mfi) {
                auto a=mf.const_array(mfi); auto box=mfi.validbox();
                for(int plane=0;plane<planes;++plane) {
                    int h=axes[plane][0],v=axes[plane][1],f=axes[plane][2];
                    Real height=plane==3 ? peak_position[2] : 0;
                    int fixed=static_cast<int>(std::floor((height-lo[f])/dx[f]));
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
            <<",\"planes\":[\"xz\",\"xy\",\"yz\""<<(peak_xy ? ",\"xy_peak\"" : "")<<"]"
            <<",\"peak_xy\":"<<(peak_xy ? "true" : "false")
            <<",\"peak_speed\":"<<peak_speed<<",\"peak_position\":["<<peak_position[0]<<","<<peak_position[1]<<","<<peak_position[2]<<"]"
            <<",\"peak_level\":"<<peak_level
            <<",\"peak_policy\":\"active composite cells; exact ties prefer smallest absolute z, then positive z, then y/x; all-zero field uses z=0\""
            <<",\"sampling\":\"finest containing cell; positive-side cell at plane boundaries; no spatial or temporal smoothing\"}\n";
    }
    amrex::Finalize();
}
