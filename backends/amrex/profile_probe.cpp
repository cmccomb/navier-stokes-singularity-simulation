#include "paper_profile.H"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::runtime_error("usage: profile_probe TABLE < points");
        ns_paper::Profile profile(argv[1]);
        std::cout<<std::setprecision(17);
        double t,dx,x,y,z;
        while (std::cin>>t>>dx>>x>>y>>z) {
            auto a=profile.potentials(x,y,z,t,dx);
            for (auto const& v:a) for (double q:v) std::cout<<q<<" ";
            std::cout<<"\n";
        }
        if (!std::cin.eof()) throw std::runtime_error("invalid point stream");
    } catch (std::exception const& e) { std::cerr<<e.what()<<"\n"; return 2; }
}
