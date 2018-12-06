/// Read the program and determin what the program does

#include <iostream>

int main() {
  int sum{0};
  int count{};
  int x;
  while (std::cin >> x) {
    sum += x;
    count += 1;
  }

  if (count != 0) {
    std::cout << "average = " << sum / count << '\n';
  }
}
