#include <iostream>
using namespace std;

struct Tacka {
    double x;
    double y;
    double z = 1;
};

Tacka vek_proizv(Tacka A, Tacka B){
    Tacka rez;
    rez.x = A.y*B.z - A.z*B.y;
    rez.y = A.z*B.x - A.x*B.z;
    rez.z = A.x*B.y - A.y*B.x;

    return rez;
}

Tacka osmoteme(Tacka A, Tacka B, Tacka C, Tacka A1, Tacka B1, Tacka C1, Tacka D1){
    
    Tacka CB = vek_proizv(C, B);
    Tacka C1B1 = vek_proizv(C1, B1);
    Tacka Xb = vek_proizv(CB, C1B1);
    
    ////////////////////////
    
    Tacka B1A1 = vek_proizv(B1, A1);
    Tacka C1D1 = vek_proizv(C1, D1);
    Tacka Yb = vek_proizv(B1A1, C1D1);
    
    //////////////////////////
    
    Tacka XbA = vek_proizv(Xb, A);
    Tacka CYb = vek_proizv(C, Yb);
    Tacka D = vek_proizv(XbA, CYb);
    
    return D;
}
int main() {
    Tacka A, B, C, D;
    Tacka A1, B1, C1, D1;
    
    A1.x = 264; A1.y = 295;
    B1.x = 526; B1.y = 482;
    C1.x = 958; C1.y = 203;
    D1.x = 726; D1.y = 112;
    
    A.x = 333; A.y = 537;
    B.x = 530; B.y = 706;
    C.x = 876; C.y = 418;
    
    D = osmoteme(A, B, C, A1, B1, C1, D1);
    cout << D.x/D.z << ", " << D.y/D.z << endl;
    
    return 0;
}
