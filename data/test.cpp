#include <iostream>
#include <iomanip>
using namespace std;

std::string reverse_string(const std::string& str);

int main() {

    std::string input = "Hello, World!";
    std::string reversed = reverse_string(input);

    cout << reversed << endl;

    return 0;

}

std::string reverse_string(const std::string& str) {
    //-- Write your code below this line 
    string reversed {};
    int size = str.size() - 1;

    for (int i {size}; i >= 0; i--) {

        reversed += str[i];

    }
    
    return reversed;
    //-- Write your code above this line
}
