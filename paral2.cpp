#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <vector>


struct Config {
    static constexpr int N = 320;
    static constexpr int a = 10e5;
    static constexpr double epsilon = 10e-8;
    static constexpr int D_x = 2;
    static constexpr int D_y = 2;
    static constexpr int D_z = 2;
    static constexpr int x_0 = -1;
    static constexpr int y_0 = -1;
    static constexpr int z_0 = -1;
    static constexpr int MPI_Tag = 123;

    static constexpr double h_x = D_x / (double)(N - 1);
    static constexpr double h_y = D_y / (double)(N - 1);
    static constexpr double h_z = D_z / (double)(N - 1);

    static constexpr double squaredH_x = h_x * h_x;
    static constexpr double squaredH_y = h_y * h_y;
    static constexpr double squaredH_z = h_z * h_z;

    int procNum = 0;
    int procRank = 0;
};

double phi(double x, double y, double z) {
    return (x * x) + (y * y) + (z * z);
}

double ro(double x, double y, double z) {
    return 6 - Config::a * phi(x, y, z);
}

double calculateX(int i) {
    return Config::x_0 + (i * Config::h_x);
}

double calculateY(int j) {
    return Config::y_0 + (j * Config::h_y);
}

double calculateZ(int k) {
    return Config::z_0 + (k * Config::h_z);
}

int calculateIndex(int i, int j, int k) {
    return i * Config::N * Config::N + j * Config::N + k;
}

void initializePhi(int layerHeight, std::vector<double>& currentLayer, Config& config) {
    for (int i = 0; i < layerHeight + 2; i++) {
        int relativeZ = i + ((config.procRank * layerHeight) - 1);
        double z = calculateZ(relativeZ);

        for (int j = 0; j < Config::N; j++) {
            double x = calculateX(j);

            for (int k = 0; k < Config::N; k++) {
                double y = calculateY(k);

                if (k != 0 && k != Config::N - 1 &&
                    j != 0 && j != Config::N - 1 &&
                    z != Config::z_0 && z != Config::z_0 + Config::D_z) {
                    currentLayer[calculateIndex(i, j, k)] = 0;
                }
                else {
                    currentLayer[calculateIndex(i, j, k)] = phi(x, y, z);
                }

            }
        }
    }
}

void printCube(double* A) {
    for (int i = 0; i < Config::N; i++) {
        for (int j = 0; j < Config::N; j++) {
            for (int k = 0; k < Config::N; k++) {
                printf(" %7.4f", A[calculateIndex(i, j, k)]);
            }
            std::cout << ";";
        }
        std::cout << std::endl;
    }
}

double calculateDelta(std::vector<double>& omega) {
    auto deltaMax = DBL_MIN;
    double x, y, z;
    for (int i = 0; i < Config::N; i++) {
        x = calculateX(i);
        for (int j = 0; j < Config::N; j++) {
            y = calculateY(j);
            for (int k = 0; k < Config::N; k++) {
                z = calculateZ(k);
                deltaMax = std::max(deltaMax, std::abs(omega[calculateIndex(i, j, k)] - phi(x, y, z)));
            }
        }
    }

    return deltaMax;
}

double updateLayer(int relativeZCoordinate, int layerIndex, std::vector<double>& currentLayer,
    std::vector<double>& currentLayerBuf) {
    int absoluteZCoordinate = relativeZCoordinate + layerIndex;
    double deltaMax = DBL_MIN;
    double x, y, z;

    if (absoluteZCoordinate == 0 || absoluteZCoordinate == Config::N - 1) {
        memcpy(currentLayerBuf.data() + layerIndex * Config::N * Config::N,
            currentLayer.data() + layerIndex * Config::N * Config::N,
            Config::N * Config::N * sizeof(double));
        deltaMax = 0;
    }
    else {
        z = calculateZ(absoluteZCoordinate);

        for (int i = 0; i < Config::N; i++) {
            x = calculateX(i);
            for (int j = 0; j < Config::N; j++) {
                y = calculateY(j);

                if (i == 0 || i == Config::N - 1 || j == 0 || j == Config::N - 1) {
                    currentLayerBuf[calculateIndex(layerIndex, i, j)] = currentLayer[calculateIndex(layerIndex, i, j)];
                }
                else {
                    currentLayerBuf[calculateIndex(layerIndex, i, j)] =
                        ((currentLayer[calculateIndex(layerIndex + 1, i, j)] + currentLayer[calculateIndex(layerIndex - 1, i, j)]) /
                            Config::squaredH_z
                            +
                            (currentLayer[calculateIndex(layerIndex, i + 1, j)] + currentLayer[calculateIndex(layerIndex, i - 1, j)]) /
                            Config::squaredH_x
                            +
                            (currentLayer[calculateIndex(layerIndex, i, j + 1)] + currentLayer[calculateIndex(layerIndex, i, j - 1)]) /
                            Config::squaredH_y
                            -
                            ro(x, y, z)) /
                        (2 / Config::squaredH_x + 2 / Config::squaredH_y +
                            2 / Config::squaredH_z +
                            Config::a);

                    if (std::abs(currentLayerBuf[calculateIndex(layerIndex, i, j)] - currentLayer[calculateIndex(layerIndex, i, j)]) > deltaMax) {
                        deltaMax = currentLayerBuf[calculateIndex(layerIndex, i, j)] - currentLayer[calculateIndex(layerIndex, i, j)];
                    }
                }
            }
        }
    }
    return deltaMax;
}

int main(int argc, char* argv[]) {
    Config config = Config();
    std::vector<double> omega;
    double deltaForTerminationCriterion = DBL_MAX;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &config.procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &config.procRank);
    MPI_Request req[4];

    if ((long)Config::N * Config::N * Config::N > INT32_MAX) {
        std::cerr << "Grid size N = " << Config::N << " is too big." << std::endl;
        return 1;
    }

    if (Config::N % config.procNum && config.procRank == 0) {
        std::cerr << "Grid size N = " << Config::N << " should be divisible by the procNum = "
            << config.procNum << std::endl;
        return 1;
    }


    int layerSize = Config::N / config.procNum;
    int layerZCoordinate = config.procRank * layerSize - 1;

    int extendedLayerSize = (layerSize + 2) * Config::N * Config::N;
    std::vector<double> currentLayer = std::vector<double>(extendedLayerSize);
    std::vector<double> currentLayerBuf = std::vector<double>(extendedLayerSize);

    initializePhi(layerSize, currentLayer, config);

    double startTime = MPI_Wtime();
    do {
        double procMaxDelta = DBL_MIN;
        double tmpMaxDelta;

        if (config.procRank != 0) {
            //отправка нижнего слоя для rank-1
            MPI_Isend(currentLayerBuf.data() + Config::N * Config::N,
                Config::N * Config::N,
                MPI_DOUBLE,
                config.procRank - 1,
                Config::MPI_Tag,
                MPI_COMM_WORLD,
                &req[1]);
            //получаем верхний слой от rank-1
            MPI_Irecv(currentLayerBuf.data(),
                Config::N * Config::N,
                MPI_DOUBLE,
                config.procRank - 1,
                Config::MPI_Tag,
                MPI_COMM_WORLD,
                &req[0]);
        }

        if (config.procRank != config.procNum - 1) {
            //отправка нижнего слоя для rank+1
            MPI_Isend(currentLayerBuf.data() + Config::N * Config::N * layerSize,
                Config::N * Config::N,
                MPI_DOUBLE,
                config.procRank + 1,
                Config::MPI_Tag,
                MPI_COMM_WORLD,
                &req[3]);
            //получаем верхний слоя от rank+1
            MPI_Irecv(currentLayerBuf.data() + Config::N * Config::N * (layerSize + 1),
                Config::N * Config::N,
                MPI_DOUBLE,
                config.procRank + 1,
                Config::MPI_Tag,
                MPI_COMM_WORLD,
                &req[2]);
        }
        //выполняется вычисление остальных точек подобласти
        for (int layerIndex = 2; layerIndex < layerSize; ++layerIndex) {
            tmpMaxDelta = updateLayer(layerZCoordinate, layerIndex, currentLayer, currentLayerBuf);
            procMaxDelta = std::max(procMaxDelta, tmpMaxDelta);
        }
        //ожидание завершения обменов
        if (config.procRank != config.procNum - 1) {
            MPI_Wait(&req[2], MPI_STATUS_IGNORE);
            MPI_Wait(&req[3], MPI_STATUS_IGNORE);
        }

        if (config.procRank != 0) {
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }

        tmpMaxDelta = updateLayer(layerZCoordinate, 1, currentLayer, currentLayerBuf);
        procMaxDelta = std::max(procMaxDelta, tmpMaxDelta);

        tmpMaxDelta = updateLayer(layerZCoordinate, layerSize, currentLayer, currentLayerBuf);
        procMaxDelta = std::max(procMaxDelta, tmpMaxDelta);

        currentLayer = currentLayerBuf;

        MPI_Allreduce(&procMaxDelta,
            &deltaForTerminationCriterion,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            MPI_COMM_WORLD);
        std::cout << deltaForTerminationCriterion << std::endl;
    } while (deltaForTerminationCriterion > Config::epsilon);

    double endTime = MPI_Wtime();

    if (config.procRank == 0) {
        omega = std::vector<double>(
            Config::N * Config::N * Config::N);
    }

    MPI_Gather(currentLayer.data() + Config::N * Config::N,
        layerSize * Config::N * Config::N,
        MPI_DOUBLE,
        omega.data(),
        layerSize * Config::N * Config::N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    if (config.procRank == 0) {
        std::cout << "Time taken: " << endTime - startTime << " s" << std::endl;
        std::cout << "Delta: " << calculateDelta(omega) << std::endl;
        std::cout << "Number of processes: " << config.procNum << std::endl;
    }
    MPI_Finalize();
    return 0;
}