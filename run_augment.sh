for((i=0; i<100; i=i+1));
do
        run_cmd="python binary_augment.py --name test_bpe0_six --file random_darts_architecture.txt --bpe 1 gpus 0,1,2,3,4,5,6,7"
        echo ${run_cmd}
        echo ${i}
        $run_cmd
done