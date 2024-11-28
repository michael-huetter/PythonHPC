import numpy as np
import numba
from numba_progress import ProgressBar
import matplotlib.pyplot as plt

@numba.njit(cache=True)
def getNeighbors(array:np.ndarray, x:int, y:int, N:int) -> None:
    return (
        array[(x + 1) % N, y] +
        array[(x - 1 + N) % N, y] +
        array[x, (y + 1) % N] +
        array[x, (y - 1 + N) % N]
    )

@numba.njit(cache=True)
def getE(array:np.ndarray, N:int) -> float:
    en = 0.0
    for y in range(N):
        for x in range(N):
            en += getNeighbors(array, x, y, N) * array[x, y]  
    return en / 2.0

@numba.njit(cache=True)
def metropolis_step(spin_lattice:np.ndarray, T:float, L:int) -> None:
        
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    
    dE = 2*spin_lattice[i,j]*getNeighbors(spin_lattice, i, j, L)
            
    if np.random.random() < np.exp(-dE/T):
        spin_lattice[i, j] *= -1
        
@numba.njit(nogil=True, cache=True)
def MCMC(spin_lattice:np.ndarray, T:float, MC_STEPS:int, L:int, progress_hook:ProgressBar) -> tuple:

    E_trace = []
    M_trace = []

    for step in range(MC_STEPS):
        metropolis_step(spin_lattice, T, L)
        if step % 30 == 0:
            E_trace.append(getE(spin_lattice, L))
            M_trace.append(np.sum(spin_lattice) / (L * L))
        progress_hook.update(1)

    return np.array(E_trace), np.array(M_trace)



if __name__ == "__main__":

    L = 150
    MC_STEPS = 5_000_000
    T_values = [0.001, 2.269, 1000]

    for T in T_values:
        spin_lattice = np.ones((L,L)) #np.random.choice([-1, 1], (L, L)) 

        with ProgressBar(total=MC_STEPS) as numba_progress:
            E_trace, M_trace = MCMC(spin_lattice, T, MC_STEPS, L, numba_progress)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].plot(E_trace)
        axs[0, 0].set_title('Energy')
        axs[1, 0].plot(M_trace)
        axs[1, 0].set_title('Magnetization')
        axs[0, 1].imshow(spin_lattice, cmap='viridis')
        axs[0, 1].set_title('Spin Lattice')
        fig.delaxes(axs[1, 1])
        plt.tight_layout()
        plt.show()        

