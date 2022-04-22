/// Sor thte standard input alphabetically.
/// Read lines of text, sort them, and print the results to the standard output.
/// if the command line names a file, read from that file. Otherwise, read from
/// the standard input. The entire input is stored in memory, so don't try
/// this with input files that exceed available RAM.
///
/// Comparison uses a locale named on the command line, or the default, unamed
/// locale if no locale is named on the command line.

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

template<class C>
struct text : std::basic_string<C> {
    text() : text{""} {}
    text(char const* s) : std::basic_string<C>(s) {}
    text(text&&) = default;  text(text const&) = default;
    text& operator=(text const&) = default;
    text& operator=(text&&) = default;
};

/// read lines of text from @p in to @p iter. Lines are appended to @p iter.
/// @param in the input stream
/// @param iter an output iterator
template<class Ch>
auto read(std::basic_istream<Ch>& in) -> std::vector<text<Ch>> {
    std::vector<text<Ch>> result;
    text<Ch> line;
    while (std::getline(in, line)) {
        result.push_back(line);
    }

    return result;
}

/// main program
int main (int argc, char* argv[]) {
    try {
        // throw an exception if an unrecoverable input error occurs, e.g.,
        // disk failure
        std::cin.exceptions(std::ios_base::badbit);

        // part 1. Read the entire input into text. If the command line names a file, 
        // read that file. Otherwise, read the standard input
        std::vector<text<char>> text; // store the lines of text here
        if (argc < 2) {
            text = read(std::cin);
        } else {
            std::ifstream in(argv[1]);
            if (not in) {
                std::perror(argv[1]);
                return EXIT_FAILURE;
            }
            text = read(in);
        }

        // part 2. Sort the text The second command line argument, if present
        // names a locale to control the sort order. Without a command line
        // argument, use the default local (which is obtained from the OS).
        std::locale const& loc{ std::locale(argc >= 3 ? argv[2] : "") };
        std::collate<char> const& collate(std::use_facet<std::collate<char>>(loc));
        std::sort(text.begin(), text.end(),
            [&collate](std::string const& a, std::string const& b){
                return collate.compare(a.data(), a.data()+a.size(),
                                       b.data(), b.data()+b.size()) < 0;
            });

        // part 3 Print the sorted text
        for (auto const& line: text) {
            std::cout << line << '\n';
        }
    } catch (std::exception& ex) {
        std::cerr << "Caught exception: " << ex.what() << '\n';
        std::cerr << "Terminating program.\n";
        std::exit(EXIT_FAILURE);
    } catch(...) {
        std::cerr << "Caught unknown exceptoin type.\nTerminating program.\n";
        std::exit(EXIT_FAILURE);
    }
}