for((i=0; i<100; i=i+1));
do
        run_cmd="python3 -m torch.distributed.launch --nproc_per_node=8  binary_augment.py --name test_bpe0_six --file random_darts_architecture.txt --bpe 1 --gpus all"
        echo ${run_cmd}
        echo ${i}
        $run_cmd
done