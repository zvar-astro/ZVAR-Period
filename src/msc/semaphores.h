#ifndef SEMAPHORES_H
#define SEMAPHORES_H

#include <semaphore.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

// const char* SEMAPHORE_NAME = "/gpu_lock";

// Function to open a semaphore
sem_t* openSemaphore(const char* name);

// Function to acquire a semaphore
void acquireSemaphore(sem_t* semaphore);

// Function to release a semaphore
void releaseSemaphore(sem_t* semaphore);

// Function to close a semaphore
void closeSemaphore(sem_t* semaphore, const char* name);

#endif // SEMAPHORES_H