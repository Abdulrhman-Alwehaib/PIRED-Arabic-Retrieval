from tokenizers import Regex, decoders, normalizers, pre_tokenizers

METASPACE = chr(0x2581)
TATWEEL = chr(0x0640)
ALEF = chr(0x0627)
ALEF_MAQSURA, YEH = chr(0x0649), chr(0x064A)
TEH_MARBUTA, HEH = chr(0x0629), chr(0x0647)
TASHKEEL = f"{chr(0x064B)}-{chr(0x0652)}{chr(0x0670)}"
ALEF_VARIANTS = "".join(map(chr, (0x0622, 0x0623, 0x0625, 0x0671)))
INVISIBLE = "".join([chr(0x00AD), chr(0x061C), chr(0x200B), "-", chr(0x200F), chr(0x202A), "-", chr(0x202E),
                     chr(0x2060), "-", chr(0x2064), chr(0x2066), "-", chr(0x2069), chr(0xFEFF)])
LINE_BREAKS = "\r\n?|[" + "".join(map(chr, (0x0B, 0x0C, 0x85, 0x2028, 0x2029))) + "]"
HORIZONTAL_SPACE = ("[\t " + chr(0x00A0) + chr(0x1680) + chr(0x2000) + "-" + chr(0x200A) + chr(0x202F) + chr(0x205F)
                    + chr(0x3000) + "]+")


def build_normalizer(cfg):
    steps = [normalizers.Replace(METASPACE, " ")]
    if cfg.nfkc:
        steps.append(normalizers.NFKC())
    if cfg.remove_invisible:
        steps.append(normalizers.Replace(Regex(f"[{INVISIBLE}]"), ""))
    if cfg.remove_tatweel:
        steps.append(normalizers.Replace(TATWEEL, ""))
    if cfg.remove_tashkeel:
        steps.append(normalizers.Replace(Regex(f"[{TASHKEEL}]"), ""))
    if cfg.unify_alef:
        steps.append(normalizers.Replace(Regex(f"[{ALEF_VARIANTS}]"), ALEF))
    if cfg.unify_alef_maqsura:
        steps.append(normalizers.Replace(ALEF_MAQSURA, YEH))
    if cfg.unify_teh_marbuta:
        steps.append(normalizers.Replace(TEH_MARBUTA, HEH))
    if cfg.arabic_digits_to_ascii:
        for i in range(10):
            steps.append(normalizers.Replace(chr(0x0660 + i), str(i)))
            steps.append(normalizers.Replace(chr(0x06F0 + i), str(i)))
    if cfg.normalize_whitespace:
        steps.append(normalizers.Replace(Regex(LINE_BREAKS), "\n"))
        steps.append(normalizers.Replace(Regex(HORIZONTAL_SPACE), " "))
        steps.append(normalizers.Replace(Regex(" ?\n[\n ]*"), "\n "))
    steps.append(normalizers.Strip())
    return normalizers.Sequence(steps)


def build_pre_tokenizer(cfg):
    steps = [
        pre_tokenizers.Metaspace(replacement=METASPACE, prepend_scheme="always", split=True),
        pre_tokenizers.Split(Regex("\n+"), behavior="isolated"),
    ]
    digit_pattern = METASPACE + r"?\p{N}" if cfg.individual_digits else METASPACE + r"?\p{N}+"
    steps.append(pre_tokenizers.Split(Regex(digit_pattern), behavior="isolated"))
    if cfg.split_punctuation:
        steps.append(pre_tokenizers.Split(Regex(METASPACE + r"?[^\p{L}\p{M}\p{N}\s" + METASPACE + "]+"),
                                          behavior="isolated"))
    return pre_tokenizers.Sequence(steps)


def build_decoder():
    return decoders.Sequence([
        decoders.Replace(METASPACE, " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(content=" ", left=1, right=0),
    ])
