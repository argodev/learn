
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
#include <mpi.h>

namespace po = boost::program_options;
namespace logging = boost::log;

uint8_t* A;
uint8_t* x;
uint8_t* my_A;
int* Ax;
int* my_Ax;


int main(int argc, const char *argv[]) {
    unsigned int totalCores = std::thread::hardware_concurrency();
    double start, finish;
    int seed = 0;
    int mod = 10;
    int dim_size = 5;
    int comm_sz;    // number of processes
    int my_rank;    // my process rank

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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
                seed = vm["seed"].as<int>();
            }
            if (vm.count("mod")) {
                mod = vm["mod"].as<int>();
            }
            if (vm.count("size")) {
                dim_size = vm["size"].as<int>();
            }
        }
    }
    catch (const po::error &ex)
    {
        std::cerr << ex.what() << '\n';
    }

    // allocating x
    x = (uint8_t*)malloc(sizeof(uint8_t)*dim_size);
    int total_size = dim_size * dim_size;
    my_A = (uint8_t*)malloc(sizeof(uint8_t)*total_size/comm_sz);
    int rank_dim_size = dim_size/comm_sz;

    if (my_rank == 0) {
        // ensure we can split evenly
        if (dim_size % comm_sz != 0) {
            BOOST_LOG_TRIVIAL(error) << "Size must be evenly divisible by the number of requested processors";
            BOOST_LOG_TRIVIAL(error) << "Size: " << dim_size << " Processors Requested: " << comm_sz;
            BOOST_LOG_TRIVIAL(error) << "Exiting.";
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        BOOST_LOG_TRIVIAL(info) << "Matrix-Vector Multiplication (MPI)";
        BOOST_LOG_TRIVIAL(info) << "Cores available:  " << totalCores;
        BOOST_LOG_TRIVIAL(info) << "MPI World Size:  " << comm_sz;
        BOOST_LOG_TRIVIAL(info) << "MPI Process Rank:  " << my_rank;
        BOOST_LOG_TRIVIAL(info) << "Using Seed Value: " << seed;
        BOOST_LOG_TRIVIAL(info) << "Using Mod Value: " << mod;
        BOOST_LOG_TRIVIAL(info) << "Using Size Value: " << dim_size;

        BOOST_LOG_TRIVIAL(info) << "Creating Matrix...";
        A = (uint8_t*)malloc(sizeof(uint8_t)*total_size);
        int current = seed;
        for (int i = 0; i < total_size; ++i) {
            current++;
            A[i] = current % mod;
        }

        BOOST_LOG_TRIVIAL(info) << "Creating Vector...";
        current = seed;
        for (int i = 0; i < dim_size; ++i) {
            current++;
            x[i] = current % mod;
        }

        // prep the answer/result
        Ax = (int*)malloc(sizeof(int)*dim_size);
        for (int i = 0; i < dim_size; ++i) {
            Ax[i] = 0;
        }

        GET_TIME(start);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(x, dim_size, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, total_size/comm_sz, MPI_UINT8_T, my_A, total_size/comm_sz, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    my_Ax = (int*)malloc(sizeof(int)*rank_dim_size);

    for (int i = 0; i < rank_dim_size; ++i) {
        int cell_ans = 0;
        for (int j = 0; j < dim_size; ++j) {
            cell_ans += (my_A[(i*dim_size)+j] * x[j]);
        }
        my_Ax[i] = cell_ans;
    }

    if (my_rank != 0) {
        // send my_Ax to rank 0
        MPI_Send(my_Ax, rank_dim_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        // update Ax with rank 0 data (local)
        for (int i = 0; i < rank_dim_size; ++i) {
            Ax[i] = my_Ax[i];
        }

        // receive from each & append/update Ax
        for (int i = 1; i < comm_sz; ++i) {
            MPI_Recv(my_Ax, rank_dim_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int j = 0; j < rank_dim_size; ++j) {
                Ax[(i*rank_dim_size) + j] = my_Ax[j];
            }
        }

        // sum
        BOOST_LOG_TRIVIAL(info) << "Summing Results...";
        long summed_answer = 0;
        for (int i = 0; i < dim_size; ++i) {
            summed_answer += Ax[i];
        }

        GET_TIME(finish);

        // be done.
        BOOST_LOG_TRIVIAL(info) << "Displaying Results:";
        BOOST_LOG_TRIVIAL(info) << "Summed Result: " << summed_answer;
        BOOST_LOG_TRIVIAL(info) << "Time Elapsed...";
        printf("Elapsed time = %e seconds\n", finish - start);
    }

    // clean up
    free(A);
    free(my_A);
    free(x);
    free(Ax);
    free(my_Ax);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
