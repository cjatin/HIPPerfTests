#include <vector>
#include <iostream>

using namespace std;

int main() {
    vector<int> v;
    v.reserve(10);
    cout << v.size() << " " << v.capacity() << endl;
}
