
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <fstream>
#include <locale>
#include <vector>
#include <thread>
#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
//#include <boost/log/utility/setup/file.hpp>
#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include "timer.h"

namespace po = boost::program_options;
namespace logging = boost::log;

uint8_t* A;
uint8_t* x;
int* Ax;


int main(int argc, const char *argv[]) {
    unsigned int totalCores = std::thread::hardware_concurrency();
    double start, finish;
    int seed = 0;
    int mod = 10;
    int dim_size = 5;
    BOOST_LOG_TRIVIAL(info) << "Matrix-Vector Multiplication (Serial)";
    BOOST_LOG_TRIVIAL(info) << "Cores available:  " << totalCores;

    // Set log level
	logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::severity_level::info);

    // let's get some of the data from the users
    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Show the usage/help screen")
            ("seed,s", po::value<int>()->default_value(0), "Seed used to fill matrix and array")
            ("mod,m", po::value<int>()->default_value(10), "Mod used to bound upper size of values in matrix/array")
            ("size,n", po::value<int>()->default_value(5), "Size length of matrix sizes and vector length");


        po::positional_options_description pos;
        pos.add("seed", 1);
        pos.add("mod", 1);
        pos.add("size", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << '\n';
            return EXIT_SUCCESS;
        }
        else {
            if (vm.count("seed")){
                BOOST_LOG_TRIVIAL(info) << "Using Seed Value: " << vm["seed"].as<int>();
                seed = vm["seed"].as<int>();
            }
            if (vm.count("mod")) {
                BOOST_LOG_TRIVIAL(info) << "Using Mod Value: " << vm["mod"].as<int>();
                mod = vm["mod"].as<int>();
            }
            if (vm.count("size")) {
                BOOST_LOG_TRIVIAL(info) << "Using Size Value: " << vm["size"].as<int>();
                dim_size = vm["size"].as<int>();
            }
        }
    }
    catch (const po::error &ex)
    {
        std::cerr << ex.what() << '\n';
    }






    BOOST_LOG_TRIVIAL(info) << "Creating Matrix...";
    int total_size = dim_size * dim_size;
    A = (uint8_t*)malloc(sizeof(uint8_t)*total_size);
    int current = seed;
    for (int i = 0; i < total_size; ++i) {
        current++;
        A[i] = current % mod;
    }
    // std::cout << "Printing Matrix (SOURCE):" << std::endl;
    // for (int i = 0; i < dim_size; ++i) {
    //     for (int j = 0; j < dim_size; ++j) {
    //         std::cout << (int)A[(i*dim_size)+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }



    BOOST_LOG_TRIVIAL(info) << "Creating Vector...";
    x = (uint8_t*)malloc(sizeof(uint8_t)*dim_size);
    current = seed;
    for (int i = 0; i < dim_size; ++i) {
        current++;
        x[i] = current % mod;
    }

    // std::cout << "Printing Vector (SOURCE):" << std::endl;
    // for (int i = 0; i < dim_size; ++i) {
    //     std::cout << (int)x[i] << std::endl;
    // }

    BOOST_LOG_TRIVIAL(info) << "Performing Muliplication...";
    GET_TIME(start);
    Ax = (int*)malloc(sizeof(int)*dim_size);
    for (int i = 0; i < dim_size; ++i) {
        int cell_ans = 0;
        for (int j = 0; j < dim_size; ++j) {
            cell_ans += (A[(i*dim_size)+j] * x[j]);
        }
        Ax[i] = cell_ans;
    }

    BOOST_LOG_TRIVIAL(info) << "Summing Results...";
    long summed_answer = 0;
    for (int i = 0; i < dim_size; ++i) {
        summed_answer += Ax[i];
    }

    GET_TIME(finish);
    BOOST_LOG_TRIVIAL(info) << "Displaying Results:";
    std::cout << "Printing Vector (Result):" << std::endl;
    for (int i = 0; i < dim_size; ++i) {
        std::cout << (int)Ax[i] << std::endl;
    }

    BOOST_LOG_TRIVIAL(info) << "Summed Result: " << summed_answer;

    BOOST_LOG_TRIVIAL(info) << "Time Elapsed...";
    printf("Elapsed time = %e seconds\n", finish - start);

    // clean up
    free(A);
    free(x);
    free(Ax);

    return EXIT_SUCCESS;
}
