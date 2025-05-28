# CIFAR10 Benchmark for FLUTE

This setup was based from https://github.com/microsoft/msrflute/tree/main/experiments/cv_cnn_femnist and updated with CIFAR10 dataset and the 2 layer CNN model.

This does not work on Mac M1...
FLUTE has to be run from the actual FLUTE repo root dir, because it references files on a relative path from that dir. That is why the CIFAR10 code is in `msrflute/experiments/cifar10`.

## setup

Download data with pfl-research:
```
(
  cd ../../../../benchmarks/
  python -m dataset.cifar10.download_preprocess --output_dir ../publications/pfl/flute/cifar10/data/cifar10
)
```
See `./setup.sh` for setup.

## Run

```code
cd msrflute
PYTHONPATH=./core/:.:../:../../../ python -m torch.distributed.run  --nproc_per_node=1  e2e_trainer.py -dataPath ./data -outputPath ./outputTest  -config ./experiments/cifar10/config.yaml -task cifar10 -backend nccl
```


## Regarding multi-process single-GPU

currently getting this error:

```
Thu Jan 18 22:49:39 2024 : Worker on node 1: process started
terminate called after throwing an instance of 'gloo::IoException'
  what():  [../third_party/gloo/gloo/transport/tcp/pair.cc:395] writev [240.57.193.204]:45263: Bad address
Traceback (most recent call last):
  File "/mnt/task_runtime/msrflute/e2e_trainer.py", line 261, in <module>
    run_worker(model_path, config, task, data_path, local_rank, backend)
  File "/mnt/task_runtime/msrflute/e2e_trainer.py", line 199, in run_worker
    worker.run()
  File "/mnt/task_runtime/msrflute/core/federated.py", line 502, in run
    command = _recv(command)
  File "/mnt/task_runtime/msrflute/core/federated.py", line 63, in _recv
    dist.recv(tensor=x, src=src)
  File "/opt/conda/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 1202, in recv
    pg.recv([tensor], src, tag).wait()
RuntimeError: [../third_party/gloo/gloo/transport/tcp/pair.cc:598] Connection closed by peer [240.57.193.204]:49556
```

when pinning both processes to GPU0, and running with 2 processes:
```
python -m torch.distributed.run  --nproc_per_node=2  e2e_trainer.py -dataPath ./data -outputPath ./outputTest  -config ./experiments/cifar10/config.yaml -task cifar10 -backend gloo
```
