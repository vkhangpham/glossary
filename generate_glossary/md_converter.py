from bs4 import BeautifulSoup, Comment
from cleantext import clean

clean_func = lambda x: clean(
    x,
    fix_unicode=True,  # fix various unicode errors
    to_ascii=True,  # transliterate to closest ASCII representation
    lower=False,  # lowercase text
    no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
    no_urls=False,  # replace all URLs with a special token
    no_emails=False,  # replace all email addresses with a special token
    no_phone_numbers=False,  # replace all phone numbers with a special token
    no_numbers=False,  # replace all numbers with a special token
    no_digits=False,  # replace all digits with a special token
    no_currency_symbols=False,  # replace all currency symbols with a special token
    no_punct=False,  # remove punctuations
    replace_with_punct="",  # instead of removing punctuations you may replace them
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en",  # set to 'de' for German special handling
)

UNWANTED_CODE_TAGS = [
    "code",
    "pre",
    "kbd",
    "samp",
    "var",
    "tt",
    "xmp",
    "script",
    "style",
    "textarea",
    "pre",
    "samp",
    "code",
    "var",
    "kbd",
    "tt",
    "xmp",
    "iframe",
    "object",
    "embed",
]

def wrap_text_with_p_tags(soup):

    for text in soup.find_all(string=True, recursive=True):
        if text.parent.name not in [
            "style",
            "script",
            "head",
            "title",
            "meta",
            "[document]",
        ]:
            if text.parent.find_all(recursive=True, string=False):
                text.wrap(soup.new_tag("p"))

    return soup

def traverse(t, current_path=None, output_file=[], level=0):
    if current_path is None:
        current_path = [t.name]

    for tag in t.find_all(recursive=False):
        path = "_".join(current_path + [tag.name])
        value = tag.find(string=True, recursive=False)
        if value and not ("<!--" in value and "-->" in value):
            output_file.append([path, value])
        if tag.find():
            traverse(tag, current_path + [tag.name], output_file, level=level + 1)

def extract_text_from_html(soup):
    dom_paths = []
    traverse(soup, output_file=dom_paths)

    res = []
    paths = []
    for path, value in dom_paths:
        if any([unwanted in path for unwanted in UNWANTED_CODE_TAGS]):
            continue
        value = clean_func(value)
        if not value:
            continue
        if any([f"h{num}" in path for num in range(1, 9)]):
            if len(value) <= 1:
                continue
            header_num = [num for num in range(1, 9) if f"h{num}" in path][0]
            to_append = "#" * header_num + " " + value

        elif any([tag in path for tag in ["_b_", "strong"]]) or path.endswith("_b"):
            to_append = f"**{value}**"
        elif path.endswith("li"):
            to_append = f"+ {value}"
        elif path.endswith("_a"):
            to_append = f"[{value}]()"
        else:
            to_append = value
        res.append(to_append)

        paths.append(path)
    return "\n---\n".join(res)

def extract_paragraphs_from_html(html_src: str, min_word_number=1):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_src, "lxml")

    # # Find and remove the header element
    # header = soup.find("header")
    # if header:
    #     header.extract()

    # Find and remove the nav element
    nav = soup.find("nav")
    if nav:
        nav.extract()

    # Find and remove the footer element
    footer = soup.find("footer")
    if footer:
        footer.extract()

    for s in soup.select("script"):
        s.extract()

    comments = soup.findAll(string=lambda text: isinstance(text, Comment))
    [comment.extract() for comment in comments]

    soup = wrap_text_with_p_tags(soup)

    try:
        temp = soup.main if soup.main else soup
    except Exception:
        temp = soup
    soup = temp

    res = extract_text_from_html(soup)
    splitted = [clean_func(item) for item in res.split("\n---\n")]
    splitted = [
        paragraph
        for paragraph in splitted
        if paragraph and len(paragraph.split(" ")) >= min_word_number
    ]

    return splitted


def convert_html_src_to_md(html_src: str, max_words=2500) -> str:
    splitted = extract_paragraphs_from_html(html_src)
    paragraphs = []
    word_count = 0
    for line in splitted:
        paragraphs.append(line)
        word_count += len(line.split())
        if word_count > max_words:
            break

    extracted_from_web = "\n".join(paragraphs)

    return extracted_from_web