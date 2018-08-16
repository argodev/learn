#include <iostream>
#include <limits>

int main() {
    // note that "digits" means binary digits, i.e. bits
    std::cout << "bits per bool: " << std::numeric_limits<bool>::digits << "\n";   
}