#include "semaphores.h"

// Function to open a semaphore
sem_t* openSemaphore(const char* name) {
    sem_t* semaphore = sem_open(name, O_CREAT, 0644, 1);
    if (semaphore == SEM_FAILED) {
        perror("sem_open");
        exit(EXIT_FAILURE);
    }
    return semaphore;
}

// Function to acquire a semaphore
void acquireSemaphore(sem_t* semaphore) {
    if (sem_wait(semaphore) != 0) {
        perror("sem_wait");
        exit(EXIT_FAILURE);
    }
}

// void acquireSemaphore(sem_t* semaphore) {
//     struct timespec ts;
//     if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
//         perror("clock_gettime");
//         exit(EXIT_FAILURE);
//     }
//     ts.tv_sec += 1; // Set timeout to 0.1 seconds

//     while (sem_timedwait(semaphore, &ts) == -1) {
//         if (errno == ETIMEDOUT) {
//             std::cerr << "sem_timedwait timed out" << std::endl;
//             // Handle timeout (e.g., retry, log, etc.)
//             return;
//         } else if (errno != EINTR) {
//             perror("sem_timedwait");
//             exit(EXIT_FAILURE);
//         }
//     }
// }

// Function to release a semaphore
void releaseSemaphore(sem_t* semaphore) {
    if (sem_post(semaphore) != 0) {
        perror("sem_post");
        exit(EXIT_FAILURE);
    }
}

// Function to close a semaphore
void closeSemaphore(sem_t* semaphore, const char* name) {
    if (sem_close(semaphore) != 0) {
        perror("sem_close");
        exit(EXIT_FAILURE);
    }
}