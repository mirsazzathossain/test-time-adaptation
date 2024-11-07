python test_time.py --cfg cfgs/cifar10_c/ours.yaml SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml SETTING mixed_domains
python test_time.py --cfg cfgs/cifar10_c/ours.yaml SETTING gradual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml SETTING correlated
python test_time.py --cfg cfgs/cifar10_c/ours.yaml SETTING continual_mixed_domain
python test_time.py --cfg cfgs/cifar10_c/ours.yaml SETTING mixed_domains_correlated

python test_time.py --cfg cfgs/cifar100_c/ours.yaml SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml SETTING mixed_domains
python test_time.py --cfg cfgs/cifar100_c/ours.yaml SETTING gradual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml SETTING correlated
python test_time.py --cfg cfgs/cifar100_c/ours.yaml SETTING continual_mixed_domain
python test_time.py --cfg cfgs/cifar100_c/ours.yaml SETTING mixed_domains_correlated

python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING continual CORRUPTION.DATASET imagenet_r
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING mixed_domains CORRUPTION.DATASET imagenet_r
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING gradual CORRUPTION.DATASET imagenet_r
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING continual_mixed_domain CORRUPTION.DATASET imagenet_r
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING correlated CORRUPTION.DATASET imagenet_r
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING mixed_domains_correlated CORRUPTION.DATASET imagenet_r

python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING continual CORRUPTION.DATASET imagenet_a
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING mixed_domains CORRUPTION.DATASET imagenet_a
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING gradual CORRUPTION.DATASET imagenet_a
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING continual_mixed_domain CORRUPTION.DATASET imagenet_a
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING correlated CORRUPTION.DATASET imagenet_a
python test_time.py --cfg cfgs/imagenet_others/ours.yaml SETTING mixed_domains_correlated CORRUPTION.DATASET imagenet_a

python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto,contr_t2,im_loss,differ_loss SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto,contr_t2,im_loss,differ_loss,mem_loss SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto,contr_t2,im_loss SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto,contr_t2 SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1 SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1,ce_s_t2 SETTING continual
python test_time.py --cfg cfgs/cifar10_c/ours.yaml LOSSES ce_s_t1 SETTING continual

python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto,contr_t2,im_loss,differ_loss SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto,contr_t2,im_loss,differ_loss,mem_loss SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto,contr_t2,im_loss SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto,contr_t2 SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto,mse_t2_proto SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1,contr_t2_proto SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1,ce_s_t2,ce_s_aug_t1 SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1,ce_s_t2 SETTING continual
python test_time.py --cfg cfgs/cifar100_c/ours.yaml LOSSES ce_s_t1 SETTING continual

python test_time.py --cfg cfgs/imagenet_c/ours.yaml SETTING continual CORRUPTION.NUM_EX 50000
python test_time.py --cfg cfgs/imagenet_c/ours.yaml SETTING mixed_domains CORRUPTION.NUM_EX 50000
python test_time.py --cfg cfgs/imagenet_c/ours.yaml SETTING gradual CORRUPTION.NUM_EX 50000
python test_time.py --cfg cfgs/imagenet_c/ours.yaml SETTING correlated CORRUPTION.NUM_EX 50000
python test_time.py --cfg cfgs/imagenet_c/ours.yaml SETTING continual_mixed_domain CORRUPTION.NUM_EX 50000
python test_time.py --cfg cfgs/imagenet_c/ours.yaml SETTING mixed_domains_correlated CORRUPTION.NUM_EX 50000