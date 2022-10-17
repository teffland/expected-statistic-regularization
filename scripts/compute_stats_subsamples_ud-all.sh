parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

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

declare -a stat_funcs=(
    "tail_dep_dir_freq"
)

declare -a sizes=( 50 )
declare -a seeds=( 0 1 2 )
for tb in "${fs[@]}"
do
  for size in "${sizes[@]}"
  do
    for seed in "${seeds[@]}"
    do

    train_fs=(data/ud-treebanks-v2.8/$tb/*train.conllu)
    train_f="${train_fs[0]}"
    gold_f=${train_f%.conllu}_S${size}_R${seed}.conllu
    stats_csv_path=${train_f%.conllu}_S${size}_R${seed}_ud-all_stats.csv

    echo $tb S$size R$seed
    echo ==============================
    python -m ml.cmd.compare_datasets\
    --gold-paths $gold_f\
    --out-csv-path $stats_csv_path\
    --n-bootstrap-samples 1000\
    --batch-size 8\
    --stat-funcs ${stat_funcs[@]}

done
done
done
echo ALL DONE