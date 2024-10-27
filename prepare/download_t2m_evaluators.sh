echo -e "Downloading T2M evaluators (in use by the evaluators, not by MoMo itself)"
gdown --fuzzy https://drive.google.com/file/d/1Gcp-SOHTrvDegKs-k5G8TnYsYPeQpdbh/view
rm -rf external_models/t2m

unzip t2m.zip -d external_models
echo -e "Cleaning\n"
rm t2m.zip

echo -e "Downloading done!"