from aip import AipOcr  #pip install baidu-aip


config = {
    'appId':'19490991',
    'apiKey':'poElZnNxn1kAgHvQVolFNM9S',
    'secretKey':'dBjiv8Q0VYu6tuXy5tzuwLM88sS8m0b5'
}

client = AipOcr(**config)

# 获取图像内容
def get_file_content(file):
    with open(file,'rb') as f:
        return f.read()

# 文字 to 字符
def img_to_str(image_path):
    image = get_file_content(image_path)
    result = client.handwriting(image)
    # print(result)
    if 'words_result' in result:
        return '\n'.join([w['words'] for w in result['words_result']])
