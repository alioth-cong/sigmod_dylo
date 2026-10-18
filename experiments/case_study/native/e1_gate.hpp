#include <thread>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <string>
static void e1gate(){
 const char* p=std::getenv("E1_GATE"); if(!p)return;
 {std::ofstream f(std::string(p)+".ready");f<<"ready";}
 while(!std::ifstream(std::string(p)+".go").good())std::this_thread::sleep_for(std::chrono::milliseconds(2));
}
struct E1ScopedTimer{
 double& total;std::chrono::steady_clock::time_point start;bool running=true;
 E1ScopedTimer(double& t):total(t),start(std::chrono::steady_clock::now()){}
 void stop(){if(running){total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();running=false;}}
 ~E1ScopedTimer(){stop();}
};
static int e1seed(){const char* p=std::getenv("E1_SEED");return p?std::atoi(p):17;}
