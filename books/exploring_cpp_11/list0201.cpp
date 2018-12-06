/// Read the program and determine what the program does

#include <iostream>
#include <limits>

int main() {
  // declare a variable named min, initialize it to the maximum possible value
  int min{std::numeric_limits<int>::max()};

  // declare an int variable named max, init it to the minimum possible value
  int max{std::numeric_limits<int>::min()};
  
  // declare a boolean variable named any and set it to false
  bool any{false};
  
  // declare an int 'x' - no specific init value
  int x;

  // loop over stdin until user presses ctl+d
  while (std::cin >> x) {

    // set any to true (indicate we got at least one value)
    any = true;

    // if the provided value is less than min, update min to that value
    if (x < min) {
      min = x;
    }

    // if the provided value is greater than max, update max to that value
    if (x > max) {
      max = x;
    }
  }

  // if we got any input values, display the new/current min/max
  if (any) {
    std::cout << "min = " << min << "\nmax = " << max << '\n';
  }
}
