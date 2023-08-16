results_folder="$HOME/pcl-registration/mbes-registration-data/src/20230711-new-data-results/"
for data in $(ls $results_folder)
do
    for method in $(ls $results_folder/$data)
        do
        if [[ "$data" != *".log"* ]]; then
            # print if data folder contains a file metrics.npz
            if [ -f "$results_folder/$data/$method/pred_metrics.npz" ]; then
                 echo "###########################"
                 echo "Skip $results_folder/$data/$method. Evaluation results exist!"
                 echo "###########################"
            else
            echo "data: $data"
            echo "method: $method"
            results_root="$results_folder/$folder/$data/$method"
            logname="$results_root/evaluation.log"
            echo "Evaluating $results_root..."
            echo "Logging to $logname..."
            python $HOME/pcl-registration/mbes-registration-data/src/evaluate_results.py \
                --results_root $results_root > $logname
            fi
        fi
        done
done
