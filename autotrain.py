import os
import time
import signal
from multiprocessing import Process, Manager

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# cmd=[

    # 'CUDA_VISIBLE_DEVICES={0} '+'python main.py --name usps --data_name usps --n_cluster 10 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={} '+'python main.py --name HAR --data_name har --n_cluster 6 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.5 0.01 --alpha 0.3 --sigma 0.5 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={} '+'python main.py --name Reuters-10k --data_name reuters-10k --n_cluster 4 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 8 --batch_size 500 --ratio 1.0 0.0 0.0 0.01 0.001 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={} '+'python main.py --name Pendigits --data_name pendigits --n_cluster 10 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.01 0.01 --alpha 0.9 --sigma 1.0 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={} '+'python main.py --name MNIST --data_name mnist-full --n_cluster 10 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 10 --batch_size 4000 --ratio 1.0 0.0 0.0 0.01 0.01 --alpha 1.0 --sigma 0.5 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={} '+'python main.py --name Coil100rgb --data_name coil100rgb --n_cluster 100 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 6 --batch_size 2400 --ratio 1.0 0.0 0.0 0.01 0.01 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1'
    
    
    # 'CUDA_VISIBLE_DEVICES={0} '+'python main.py --name AC --data_name AC --n_cluster 2 --pretrain 1 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha 0.1 --sigma 1 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={0} '+'python main.py --name sparse_3_dense_3_dense_3 --data_name sparse_3_dense_3_dense_3 --n_cluster 3 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={0} '+'python main.py --name sparse_8_dense_1_dense_1 --data_name sparse_8_dense_1_dense_1 --n_cluster 3 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={0} '+'python main.py --name one_gaussian_10_one_line_5_2 --data_name one_gaussian_10_one_line_5_2 --n_cluster 2 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={0} '+'python main.py --name sparse_3_dense_3_dense_3_10 --data_name sparse_3_dense_3_dense_3_10 --n_cluster 3 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 10 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={0} '+'python main.py --name sparse_8_dense_1_dense_1_10 --data_name sparse_8_dense_1_dense_1_10 --n_cluster 3 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1',
    # 'CUDA_VISIBLE_DEVICES={0} '+'python main.py --name one_gaussian_10_one_line_5_2_10 --data_name one_gaussian_10_one_line_5_2_10 --n_cluster 2 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha 1.0 --sigma 1.0 --vtrace_out 0.001 0.1',
# ]


cmd = []
alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
sigma_list = [0.5, 1.0, 3.0, 5.0, 8.0, 10.0]
for alpha in alpha_list:
    for sigma in sigma_list:
        
        cmd.append('python main.py --name AC --data_name AC --n_cluster 2 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha {alpha} --sigma {sigma} --vtrace_out 0.001 0.1'.format(alpha=alpha, sigma=sigma))
#         cmd.append('python main.py --name 4C --data_name 4C --n_cluster 4 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha {alpha} --sigma {sigma} --vtrace_out 0.001 0.1'.format(alpha=alpha, sigma=sigma))
#         cmd.append('python main.py --name sparse_3_dense_3_dense_3 --data_name sparse_3_dense_3_dense_3 --n_cluster 3 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha {alpha} --sigma {sigma} --vtrace_out 0.001 0.1'.format(alpha=alpha, sigma=sigma))
#         cmd.append('python main.py --name sparse_8_dense_1_dense_1 --data_name sparse_8_dense_1_dense_1 --n_cluster 3 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha {alpha} --sigma {sigma} --vtrace_out 0.001 0.1'.format(alpha=alpha, sigma=sigma))
#         cmd.append('python main.py --name one_gaussian_10_one_line_5_2 --data_name one_gaussian_10_one_line_5_2 --n_cluster 2 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha {alpha} --sigma {sigma} --vtrace_out 0.001 0.1'.format(alpha=alpha, sigma=sigma))
#         cmd.append('python main.py --name sparse_3_dense_3_dense_3_10 --data_name sparse_3_dense_3_dense_3_10 --n_cluster 3 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha {alpha} --sigma {sigma} --vtrace_out 0.001 0.1'.format(alpha=alpha, sigma=sigma))
#         cmd.append('python main.py --name sparse_8_dense_1_dense_1_10 --data_name sparse_8_dense_1_dense_1_10 --n_cluster 3 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha {alpha} --sigma {sigma} --vtrace_out 0.001 0.1'.format(alpha=alpha, sigma=sigma))
#         cmd.append('python main.py --name one_gaussian_10_one_line_5_2_10 --data_name one_gaussian_10_one_line_5_2_10 --n_cluster 2 --pretrain 2 --epochs 500 --log_interval 50 --perplexity 3 --batch_size 500 --ratio 1.0 0.0 0.0 0.05 0.005 --alpha {alpha} --sigma {sigma} --vtrace_out 0.001 0.1'.format(alpha=alpha, sigma=sigma))


def run(command,gpuid,gpustate):
    os.system(command.format(gpuid))
    gpustate[str(gpuid)] = True

def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes) )
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
    except Exception as e:
        print(str(e))

if __name__  == '__main__':
    signal.signal(signal.SIGTERM, term)

    gpustate = Manager().dict({str(i):True for i in range(0, 8)})
    processes = []
    idx = 0

    while idx < len(cmd):
        for gpuid in range(0, 8):
            if gpuid == 0 or gpuid == 1:
                if gpustate[str(gpuid)] == True:
                    print(idx)
                    gpustate[str(gpuid)] = False
                    p = Process(target = run, args = (cmd[idx], gpuid, gpustate), name = str(gpuid))
                    p.start()

                    print(gpustate)
                    processes.append(p)
                    idx += 1
                    # if idx % 2 == 0:
                    #     time.sleep(300.0)
                    
                    break

    for p in processes:
        p.join()
