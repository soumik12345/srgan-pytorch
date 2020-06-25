mkdir /root/.kaggle
touch /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
echo "{\"username\":\"<YOUR-USERNAME>\",\"key\":\"<YOUR-KAGGLE-KEY>\"}" > /root/.kaggle/kaggle.json
kaggle datasets download -d huanghanchina/pascal-voc-2012
unzip -q ./pascal-voc-2012.zip
rm ./pascal-voc-2012.zip