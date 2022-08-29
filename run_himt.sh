
for i in 'twitter2015'  #'twitter' 
do 
	for l in '1e-5'
	do 
		for pl in '1' 
		do
			for cl in '2'
			do
				for w in '0.1'
				do
					for seed in '3813356'
					do
					echo ${i}
					echo ${l}
					echo ${pl}
					echo ${cl}
					echo ${w}
					echo ${seed}
					CUDA_VISIBLE_DEVICES=0 python run_himt.py  \
					--data_dir ../pytorch-pretrained-absa/absa_data/${i} \
					--task_name ${i} \
					--output_dir ./output/HIMT_TopN/${i}_${l}_${pl}_${cl}_${w}_${seed}/ \
					--bert_model ../pytorch-bert-base \
					--do_train \
					--do_eval \
					--r_loss 1e-2 \
					--warmup_proportion ${w} \
					--learning_rate ${l} \
					--projectionlayer ${pl} \
					--seed ${seed} \
					--cmtlayer ${cl} \
					--train_batch_size 32 \
					--max_target_length 80 \
					--no_gate True \
					--img_path ../HIMT/data/twitter2015_top3/TSA_Twitter2015_
					done
				done
			done
		done
	done
done
for i in 'twitter'
do 
	for l in '2e-5' 
	do 
		for pl in '1'
		do
			for cl in '4'
			do
				for w in '0.1'
				do
					for seed in '3813356' 
					do
					echo ${i}
					echo ${l}
					echo ${pl}
					echo ${cl}
					echo ${w}
					echo ${seed}
					CUDA_VISIBLE_DEVICES=0 python run_himt.py  \
					--data_dir ../pytorch-pretrained-absa/absa_data/${i} \
					--task_name ${i} \
					--output_dir ./output/HIMT_TopN/temp/${i}_${l}_${pl}_${cl}_${w}_${seed}/ \
					--bert_model ../pytorch-bert-base \
					--do_train \
					--do_eval \
					--r_loss 1e-2 \
					--warmup_proportion ${w} \
					--learning_rate ${l} \
					--projectionlayer ${pl} \
					--seed ${seed} \
					--cmtlayer ${cl} \
					--train_batch_size 32 \
					--max_target_length 80 \
					--no_gate True \
					--img_path ../HIMT/data/twitter2017_top3/TSA_Twitter2017_
					done
				done
			done
		done
	done
done