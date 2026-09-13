// Lossless export of the fixed rectangular native levels; never resample a field.
#include <AMReX_PlotFileDataImpl.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_MFIter.H>
#include <AMReX_Loop.H>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

int main(int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        using namespace amrex;
        ParmParse pp;
        std::string plot, output;
        pp.get("plot", plot); pp.get("output", output);
        PlotFileDataImpl data(plot);
        Vector<std::string> names{"velx","vely","velz","forcing_x","forcing_y","forcing_z"};
        if (data.spaceDim()!=3 || data.varNames()!=names) Abort("native vectors missing");
        auto lo=data.probLo(), hi=data.probHi();
        for(int d=0;d<3;++d) if(lo[d]!=-1 || hi[d]!=1) Abort("expected full periodic box");
        static_assert(std::endian::native==std::endian::little && sizeof(Real)==8);
        std::ofstream stream(output,std::ios::binary);
        if(!stream) Abort("cannot open output");
        std::ostringstream record;
        record << std::setprecision(17) << "{\"schema_version\":1,\"time\":" << data.time()
               << ",\"dtype\":\"<f8\",\"order\":\"xyz-component\",\"levels\":[";
        Long offset=0;
        for(int lev=0;lev<=data.finestLevel();++lev) {
            auto mf=data.get(lev); auto dx=data.cellSize(lev);
            auto bounds=mf.boxArray().minimalBox();
            if(!mf.boxArray().isDisjoint() || bounds.numPts()!=mf.boxArray().numPts())
                Abort("export requires disjoint, fully tiled rectangular levels");
            if(bounds.numPts()>Long(16777216)) Abort("level exceeds bounded export memory");
            auto low=bounds.smallEnd(); auto size=bounds.length();
            std::vector<Real> values(bounds.numPts()*6,std::numeric_limits<Real>::quiet_NaN());
            for(MFIter mfi(mf);mfi.isValid();++mfi) {
                auto a=mf.const_array(mfi);
                LoopOnCpu(mfi.validbox(),[&](int i,int j,int k) {
                    Long p=((Long(i-low[0])*size[1]+j-low[1])*size[2]+k-low[2])*6;
                    for(int c=0;c<6;++c) values[p+c]=a(i,j,k,c);
                });
            }
            for(auto value:values) if(!std::isfinite(value)) Abort("incomplete/nonfinite level");
            stream.write(reinterpret_cast<char const*>(values.data()),values.size()*sizeof(Real));
            if(!stream) Abort("volume export failed");
            if(lev) record << ",";
            record << "{\"level\":" << lev << ",\"offset_bytes\":" << offset
                   << ",\"shape\":[" << size[0] << "," << size[1] << "," << size[2] << ",6]"
                   << ",\"index_lo\":[" << low[0] << "," << low[1] << "," << low[2] << "]"
                   << ",\"origin\":[" << lo[0]+low[0]*dx[0] << "," << lo[1]+low[1]*dx[1] << "," << lo[2]+low[2]*dx[2] << "]"
                   << ",\"spacing\":[" << dx[0] << "," << dx[1] << "," << dx[2] << "]}";
            offset+=values.size()*sizeof(Real);
        }
        stream.close(); if(!stream) Abort("volume close failed");
        record << "],\"bytes\":" << offset << "}";
        Print() << "NS_VOLUME_RESULT " << record.str() << "\n";
    }
    amrex::Finalize();
}
