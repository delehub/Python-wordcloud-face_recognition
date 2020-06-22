import wordcloud

w = wordcloud.WordCloud()

w.generate('AI IOT 5G Deeplearning deepfake')

w.to_file('output1.png')
