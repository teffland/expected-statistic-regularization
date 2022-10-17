export wandb_tag="UD-13 S50"
export wandb_note="Running S50 on many languages"

declare -a fs=(
    # "UD_German-HDT"
	"UD_Czech-PDT"
	"UD_Russian-SynTagRus"
	"UD_Classical_Chinese-Kyoto"
	"UD_Japanese-BCCWJ"
	"UD_Icelandic-IcePaHC"
	# "UD_Persian-PerDT"
	"UD_Estonian-EDT"
	"UD_Romanian-Nonstandard"
	"UD_Korean-Kaist"
	"UD_Belarusian-HSE"
	"UD_Latin-ITTB"
	"UD_Polish-PDB"
	"UD_Arabic-NYUAD"
	"UD_Norwegian-Bokmaal"
	"UD_Turkish-Kenet"
	"UD_Ancient_Greek-PROIEL"
	"UD_Finnish-FTB"
	"UD_French-FTB"
	"UD_Spanish-AnCora"
	"UD_Old_French-SRCMF"
	"UD_Old_East_Slavic-TOROT"
	"UD_Hindi-HDTB"
	"UD_Catalan-AnCora"
	"UD_Italian-ISDT"
	"UD_English-EWT"
	"UD_Dutch-Alpino"
	"UD_Latvian-LVTB"
	"UD_Portuguese-GSD"
	"UD_Bulgarian-BTB"
	"UD_Slovak-SNK"
	"UD_Naija-NSC"
	"UD_Croatian-SET"

	"UD_Slovenian-SSJ"
	"UD_Ukrainian-IU"
	"UD_Basque-BDT"
	"UD_Hebrew-HTB"
	# "UD_Indonesian-GSD"
	"UD_Danish-DDT"
	"UD_Swedish-Talbanken"
	"UD_Old_Church_Slavonic-PROIEL"
	"UD_Urdu-UDTB"
	"UD_Irish-IDT"
	"UD_Chinese-GSD"
	"UD_Gothic-PROIEL"
	"UD_Serbian-SET"
	"UD_Scottish_Gaelic-ARCOSG"
	"UD_Lithuanian-ALKSNIS"
	"UD_Galician-CTG"
	"UD_Armenian-ArmTDP"

	"UD_Greek-GDT"
	"UD_Uyghur-UDT"
	"UD_Hindi_English-HIENCS"
	# "UD_Vietnamese-VTB"
	"UD_Afrikaans-AfriBooms"
	"UD_Wolof-WTB"
	# "UD_Maltese-MUDT"
	"UD_Coptic-Scriptorium"
	"UD_Telugu-MTG"
	"UD_Faroese-FarPaHC"
	"UD_Hungarian-Szeged"
	"UD_Western_Armenian-ArmTDP"
	"UD_Turkish_German-SAGT"
	"UD_Welsh-CCG"
	"UD_Tamil-TTB"
	"UD_Marathi-UFAL"
)


for treebank in "${fs[@]}"
do

    export log_name=22-06-25_ud-13_S50
    export config_path=config/supervised.jsonnet

    train_fs=(data/ud-treebanks-v2.8/$treebank/*train.conllu)
    train_f="${train_fs[0]}"
    export test_data_path=${train_f%train.conllu}test.conllu
    export vocab_path=data/vocab/udify-13_vocabulary
    export weights_file_path=data/weights/udify-13-model_weights.th


    source /home/ubuntu/syntax//scripts/experiment_setup.sh
    set_base_hps
    export num_batches_per_epoch=200
    export num_epochs=100
    export scheduler_num_epochs=$num_epochs
    export patience=10
    export learning_rate=0.00002
    declare -a sizes=( 50 )
    declare -a seeds=( 0 1 2 )
    for size in "${sizes[@]}"
    do
        for seed in "${seeds[@]}"
        do

            export train_data_path=${train_f%.conllu}_S${size}_R${seed}.conllu
            # Dev size = min(train size, 100) examples
            if [ "$size" -lt 100 ]
            then
                export validation_data_path=${train_f%train.conllu}dev_S${size}_R${seed}.conllu
            else
                export validation_data_path=${train_f%train.conllu}dev_S100_R${seed}.conllu
            fi
            
            export wandb_name="${treebank}_ud-13_sup_S${size}_R${seed}"
            run_train
			# exit
            eval_dev
            # make_archive
            eval_test
            # sync_results
            cleanup_weights
        done
    done
done
echo "DONE"