# network
![kanto](./kanto_anime.gif)

- 200x200の画像に黒で壁を書いて、青でエサの点を書いたものをmakemeshに与えるとメッシュを作ります
- そのとき作ったメッシュが非連結だと動きません
    - 通路が細すぎるとメッシュがちぎれます
- makeAnimationで画像とメッシュ情報(pickle)からgifを生成

## requirements
- numpy
- matplotlib (imagemagick or pillow enabled)
- cv2

## examples

### kanto
```
python makemesh.py kanto.png --output kanto_node.pickle
python makeAnimation.py kanto.png kanto_node.pickle --output kanto_anime.gif
```

### maze
```
python makemesh.py maze.png --output maze_node.pickle
python makeAnimation.py maze.png maze_node.pickle --output maze_anime.gif --frames 20
```
## citation
```
@article{article,
    author = {Tero, Atsushi and Takagi, Seiji and Saigusa, Tetsu and Ito, Kentaro and Bebber, Daniel and Fricker, Mark and Yumiki, Kenji and Kobayashi, Ryo and Nakagaki, Toshiyuki},
    year = {2010},
    month = {01},
    pages = {439-42},
    title = {Rules for Biologically Inspired Adaptive Network Design},
    volume = {327},
    journal = {Science (New York, N.Y.)},
    doi = {10.1126/science.1177894}
}
```
