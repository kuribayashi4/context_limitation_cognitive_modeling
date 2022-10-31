import itertools
import mojimoji
import unicodedata

should_break = False
with open("data/BE/word.txt") as f:
    for bunsetsu, lines in itertools.groupby(f, key=lambda x: x.split("\t")[5]):
        if bunsetsu == "surface":
            continue
        lines = list(lines)
        if lines[0].strip().split("\t")[-1] == "B" and should_break:
            print()
        print(unicodedata.normalize("NFKC", mojimoji.han_to_zen(bunsetsu)), end="")
        should_break = True
