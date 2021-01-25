two-way branch

# 1. 원하는 감정으로 합성하기
## 1. 합성 문장을 적은 스크립트 생성 => text_text.txt
아래와 같이 스크립트를 작성한다.
첫번째, 3번째 컬럼은 플레이스홀더이다.
두번째 컬럼: 말할 내용을 적는다. 소리나는대로 써도 되고, 맞춤법대로 써도 된다.
네, 다섯번째 컬럼: 원하는 감정으로 적는다.

```
-|있잖아요, 오늘은 일월 오일 화요일이거든요.|-|3|3
-|있잖아요, 오늘은 일월 오일 화요일이거든요.|-|0|0
...
```
| No. |   감정  |
|:---:|:-------:|
|  0  | Neutral |
|  1  |   Happy |
|  2  |     Sad |
|  3  |   Angry |

## 2. 경로 설정
화자, 스크립트, 설정파일, checkpoint, output 음원의 경로를 설정한다.
inference_from_guide.py를 열어
checkpoint_path, waveglow_path, test_text_path, path를 설정한다.
화자는 nes로 하드코딩되어있는데 WaveGlow가 single speaker라서 모든 화자에 대해서 음질이 균등하지 않아 nes 화자를 추천한다.
화자에 대한 정보를 담은 excel파일은 pitchtron에 있다.

## run
```
python inference_from_guide.py
```
