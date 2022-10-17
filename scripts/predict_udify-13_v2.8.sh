# Run the udify-13 model (from v2.3) on the new languages in 2.8 with dev data
ARCHIVE=/home/ubuntu/syntax/udify/udify-13-model/model.tar.gz;


declare -a fs=(
	"UD_Arabic-PADT"
	"UD_Finnish-TDT"
	"UD_Japanese-GSD"
	"UD_Korean-GSD"
	"UD_Turkish-IMST"
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


for tb in "${fs[@]}"
do
    echo "Treebank ${tb}"
    for train_f in /home/ubuntu/syntax//data/ud-treebanks-v2.8/$tb/*train.conllu
    do
        pred_f=${train_f%.conllu}_udify-13_pred.conllu
        eval_f=${pred_f%pred.conllu}results.json
        echo
        echo "python predict.py $ARCHIVE $train_f $pred_f --eval_file $eval_f"
        ~/.pyenv/versions/udify/bin/python predict.py $ARCHIVE $train_f $pred_f --eval_file $eval_f


        dev_f=${train_f%train.conllu}dev.conllu
        pred_f=${dev_f%.conllu}_udify-13_pred.conllu
        eval_f=${pred_f%pred.conllu}results.json
        echo
        echo "python predict.py $ARCHIVE $dev_f $pred_f --eval_file $eval_f"
        ~/.pyenv/versions/udify/bin/python predict.py $ARCHIVE $dev_f $pred_f --eval_file $eval_f


        test_f=${train_f%train.conllu}test.conllu
        pred_f=${test_f%.conllu}_udify-13_pred.conllu
        eval_f=${pred_f%pred.conllu}results.json
        echo
        echo "python predict.py $ARCHIVE $test_f $pred_f --eval_file $eval_f"
        /home/ubuntu/.pyenv/versions/udify2/bin/python /home/ubuntu/syntax/udify/predict.py $ARCHIVE $test_f $pred_f --eval_file $eval_f

    done
done
echo "ALL DONE"