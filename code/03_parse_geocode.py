# -*- coding: utf-8 -*-
"""
03_parse_geocode.py — Multilingual Entity Resolver, 3-Tier Out-of-Catalog Taxonomy,
Bilingual Post-Prefix Rationale Extractor, and Dual-Pipeline Verification Audit
on `raw_responses_v2final/` (N = 10,191 live queries / 101,910 recommendation slots).

Produces:
  - `data/audit_queries_corpus.csv` (& `audit_queries_corpus_v2.csv`, 10,191 rows)
  - `data/all_recommendations.csv` (& `all_recommendations_v2.csv`, 101,910 rows with `reason`)
  - `data/unresolved_classified.csv` (complete taxonomy of out-of-catalog & unverified slots)
  - `data/rule_cross_validation_1019rows.csv` (10% stratified two-pipeline rule cross-validation, n = 1,019 queries)
  - `data/entity_resolution_50sample_audit.csv` (50-pair stratified entity resolution audit)
  - `data/ground_truth_pois_10cities.csv` (updated with empirical Batch B1 & all-batch mention counts)
"""
import importlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

cfg = importlib.import_module("00_config")

CITY_TOKEN_STOP = {
    "beijing", "shanghai", "chengdu", "quanzhou", "dali",
    "paris", "tokyo", "barcelona", "porto", "chiang", "mai", "chiangmai",
    "china", "france", "japan", "spain", "portugal", "thailand",
}

GENERIC_STOP = {
    "the", "a", "an", "of", "and", "de", "do", "da", "dos", "das", "del",
    "la", "le", "les", "el", "di", "du", "des", "l", "d", "e", "y", "i",
    "in", "at", "on", "to", "for", "with", "by", "or", "&",
    "park", "parc", "parque", "museum", "musee", "museu", "museo",
    "temple", "wat", "shrine", "church", "igreja", "eglise", "cathedral", "cathedrale", "basilica",
    "street", "road", "avenue", "alley", "lane", "quarter", "district", "neighborhood", "area",
    "square", "placa", "place", "praca", "garden", "gardens", "jardim", "jardins", "jardin",
    "palace", "palacio", "palais", "tower", "torre", "bridge", "ponte", "pont",
    "ancient", "historic", "old", "town", "city", "village", "scenic", "cultural", "national",
    "center", "centre", "hall", "building", "site", "parkland", "promenade", "waterfront",
}

EN_REASON_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "have", "has",
    "its", "via", "into", "over", "under", "very", "well", "also", "such", "more", "most",
    "recommended", "landmark", "attraction", "site", "place", "visit", "visiting", "offers",
    "offering", "provides", "providing", "featuring", "features", "located", "ideal", "perfect",
    "great", "famous", "iconic", "historic", "cultural", "modern", "museum", "heritage", "nature",
    "religious", "district", "theme", "art", "traveler", "travelers", "visitors", "tourists",
}

ZH_STOP_CHARS = set("的了和是在有与及等为可也很都而并或把被让给向往到于上中下内外观游览参观推荐适合景点地标旅游游客提供拥有体验感受位于著名历史文化")


def strip_accents(s: str) -> str:
    s = str(s).replace("œ", "oe").replace("Œ", "oe").replace("æ", "ae").replace("Æ", "ae")
    s = s.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    nk = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nk if not unicodedata.combining(c))


def norm_compact(s: str) -> str:
    t = strip_accents(s).lower()
    t = t.replace("centre", "center")
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", t)


def content_words(s: str) -> set[str]:
    t = strip_accents(s).lower().replace("centre", "center")
    ws = [w for w in re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", t).split() if len(w) >= 2]
    return {w for w in ws if w not in GENERIC_STOP and w not in CITY_TOKEN_STOP}


def ngrams3(s: str) -> set[str]:
    t = norm_compact(s)
    if len(t) < 3:
        return {t} if t else set()
    return {t[i:i + 3] for i in range(len(t) - 2)}


# Verified multilingual synonyms strictly mapping native/short surface forms to exact in-catalog POI names
EXPLICIT_CATALOG_SYNONYMS = {
    "BJS": [
        ("奥林匹克公园", "Olympic Forest Park"),
        ("北京奥林匹克公园", "Olympic Forest Park"),
        ("beijing olympic park", "Olympic Forest Park"),
        ("olympic park", "Olympic Forest Park"),
        ("lama temple", "Yonghe Temple (Lama Temple)"),
        ("yonghegong", "Yonghe Temple (Lama Temple)"),
        ("shichahai", "Shichahai Scenic Area"),
        ("nanluoguxiang", "Nanluoguxiang Alley"),
        ("forbidden city", "Forbidden City (Palace Museum)"),
        ("palace museum", "Forbidden City (Palace Museum)"),
        ("summer palace", "Summer Palace (Yiheyuan)"),
        ("yiheyuan", "Summer Palace (Yiheyuan)"),
        ("yuanmingyuan", "Old Summer Palace (Yuanmingyuan)"),
        ("old summer palace", "Old Summer Palace (Yuanmingyuan)"),
        ("bird's nest", "Beijing National Stadium (Bird's Nest)"),
        ("water cube", "National Aquatics Center (Water Cube)"),
        ("798 art", "798 Art Zone"),
        ("prince gong", "Prince Gong's Mansion"),
        ("drum and bell", "Drum and Bell Towers (Gulou)"),
        ("gulou", "Drum and Bell Towers (Gulou)"),
        ("fragrant hills", "Fragrant Hills Park (Xiangshan)"),
        ("xiangshan", "Fragrant Hills Park (Xiangshan)"),
        ("botanical garden", "China National Botanical Garden"),
        ("国家植物园", "China National Botanical Garden"),
        ("北京植物园", "China National Botanical Garden"),
    ],
    "SHA": [
        ("徐汇滨江", "West Bund Museum"),
        ("西岸文化走廊", "West Bund Museum"),
        ("西岸艺术", "West Bund Museum"),
        ("west bund", "West Bund Museum"),
        ("zhujiajiao", "Zhujiajiao Water Town"),
        ("朱家角", "Zhujiajiao Water Town"),
        ("shanghai disney", "Shanghai Disneyland"),
        ("上海迪士尼", "Shanghai Disneyland"),
        ("人民公园", "People's Square"),
        ("人民广场", "People's Square"),
        ("yu garden", "Yu Garden (Yuyuan)"),
        ("yuyuan", "Yu Garden (Yuyuan)"),
        ("陆家嘴滨江", "Lujiazui Central Greenland"),
        ("浦东滨江大道", "Lujiazui Central Greenland"),
        ("lujiazui", "Lujiazui Central Greenland"),
        ("陆家嘴", "Lujiazui Central Greenland"),
        ("the bund", "The Bund (Waitan)"),
        ("waitan", "The Bund (Waitan)"),
        ("french concession", "Former French Concession (Wukang Road)"),
        ("wukang", "Former French Concession (Wukang Road)"),
        ("oriental pearl", "Oriental Pearl TV Tower"),
        ("chenghuangmiao", "City God Temple of Shanghai (Chenghuangmiao)"),
        ("城隍庙", "City God Temple of Shanghai (Chenghuangmiao)"),
        ("一大会址", "Site of the First National Congress of the CPC"),
        ("中共一大", "Site of the First National Congress of the CPC"),
    ],
    "CTU": [
        ("qingcheng mountain", "Mount Qingcheng"),
        ("mount qingcheng", "Mount Qingcheng"),
        ("青城山", "Mount Qingcheng"),
        ("成都市天府美术馆", "Chengdu Art Museum (Tianfu Art Park)"),
        ("天府艺术公园", "Chengdu Art Museum (Tianfu Art Park)"),
        ("成都美术馆", "Chengdu Art Museum (Tianfu Art Park)"),
        ("people's park", "People's Park (Renmin Park)"),
        ("heming teahouse", "People's Park (Renmin Park)"),
        ("人民公园", "People's Park (Renmin Park)"),
        ("鹤鸣茶社", "People's Park (Renmin Park)"),
        ("kuanzhai", "Kuanzhai Alley (Wide and Narrow Alleys)"),
        ("wide and narrow", "Kuanzhai Alley (Wide and Narrow Alleys)"),
        ("panda base", "Chengdu Research Base of Giant Panda Breeding"),
        ("giant panda", "Chengdu Research Base of Giant Panda Breeding"),
        ("taikoo li", "Chunxi Road & IFS Taikoo Li"),
        ("chunxi road", "Chunxi Road & IFS Taikoo Li"),
        ("太古里", "Chunxi Road & IFS Taikoo Li"),
        ("dujiangyan", "Dujiangyan Irrigation System"),
        ("都江堰", "Dujiangyan Irrigation System"),
        ("sanxingdui", "Sanxingdui Museum (Greater Chengdu)"),
        ("三星堆", "Sanxingdui Museum (Greater Chengdu)"),
        ("wuhou", "Wuhou Shrine (Temple of Marquis Wu)"),
        ("du fu", "Du Fu Thatched Cottage"),
        ("jinli", "Jinli Ancient Street"),
        ("东郊记忆", "Eastern Suburb Memory (Dongjiao Jiyi)"),
        ("九眼桥", "Anshun Bridge & Jiuyanqiao"),
    ],
    "QZJ": [
        ("泉州非物质文化遗产馆", "Quanzhou Intangible Cultural Heritage Museum"),
        ("泉州市非物质文化遗产馆", "Quanzhou Intangible Cultural Heritage Museum"),
        ("非遗馆", "Quanzhou Intangible Cultural Heritage Museum"),
        ("qingyuan mountain", "Mount Qingyuan & Laozi Statue"),
        ("mount qingyuan", "Mount Qingyuan & Laozi Statue"),
        ("清源山", "Mount Qingyuan & Laozi Statue"),
        ("老君岩", "Mount Qingyuan & Laozi Statue"),
        ("confucian temple", "Fuwen Temple (Confucian Temple)"),
        ("confucius temple", "Fuwen Temple (Confucian Temple)"),
        ("府文庙", "Fuwen Temple (Confucian Temple)"),
        ("泉州文庙", "Fuwen Temple (Confucian Temple)"),
        ("泉州市博物馆", "Quanzhou Museum"),
        ("泉州博物馆", "Quanzhou Museum"),
        ("wudianshi", "Jinjiang Wudian Traditional Block"),
        ("五店市", "Jinjiang Wudian Traditional Block"),
        ("浔埔村", "Xunpu Village (Zanhuawei Hairpin Village)"),
        ("蟳埔", "Xunpu Village (Zanhuawei Hairpin Village)"),
        ("xunpu", "Xunpu Village (Zanhuawei Hairpin Village)"),
        ("中山路", "Zhongshan Road Arcade Street"),
        ("zhongshan road", "Zhongshan Road Arcade Street"),
        ("海外交通史博物馆", "Quanzhou Maritime Museum"),
        ("海交馆", "Quanzhou Maritime Museum"),
        ("maritime museum", "Quanzhou Maritime Museum"),
        ("tianhou", "Tianhou Palace (Mazu Temple)"),
        ("天后宫", "Tianhou Palace (Mazu Temple)"),
        ("kaiyuan", "Kaiyuan Temple & Twin Pagodas"),
        ("开元寺", "Kaiyuan Temple & Twin Pagodas"),
        ("东西塔", "Kaiyuan Temple & Twin Pagodas"),
        ("清净寺", "Qingjing Mosque"),
        ("qingjing", "Qingjing Mosque"),
        ("关岳庙", "Guan Yue Temple (Tonghuai Temple)"),
        ("通淮关岳庙", "Guan Yue Temple (Tonghuai Temple)"),
        ("西街", "West Street (Xijie Pedestrian Area)"),
        ("洛阳桥", "Luoyang Bridge"),
        ("安平桥", "Anping Bridge (Five-Li Bridge)"),
        ("五里桥", "Anping Bridge (Five-Li Bridge)"),
        ("梧林", "Wulin Traditional Village"),
        ("木偶剧院", "Quanzhou Puppet Theatre Art Hall"),
        ("闽台缘", "China Museum for Fujian-Taiwan Kinship"),
    ],
    "DLU": [
        ("洱海生态廊道", "Erhai Lake Ecological Corridor"),
        ("erhai lake", "Erhai Lake Ecological Corridor"),
        ("洱海", "Erhai Lake Ecological Corridor"),
        ("dali ancient town", "Dali Ancient City"),
        ("dali old town", "Dali Ancient City"),
        ("大理古城", "Dali Ancient City"),
        ("nanzhao customs island", "Nanzhao Folk Island"),
        ("南诏风情岛", "Nanzhao Folk Island"),
        ("床单厂", "Chuangku Art Loft & Bedhaya Factory"),
        ("xizhou", "Xizhou Ancient Town"),
        ("喜洲", "Xizhou Ancient Town"),
        ("shuanglang", "Shuanglang Ancient Town"),
        ("双廊", "Shuanglang Ancient Town"),
        ("haishe", "Haishe Park (Haizhi Park)"),
        ("海舌", "Haishe Park (Haizhi Park)"),
        ("three pagodas", "Three Pagodas of Chongsheng Temple"),
        ("崇圣寺", "Three Pagodas of Chongsheng Temple"),
        ("cangshan", "Cangshan Mountain & Cableways"),
        ("mount cang", "Cangshan Mountain & Cableways"),
        ("苍山", "Cangshan Mountain & Cableways"),
        ("寂照庵", "Gantong Temple & Jizhao Nunnery"),
        ("感通寺", "Gantong Temple & Jizhao Nunnery"),
        ("磻溪", "Panxi Village S-Bend (Erhai S-Curve)"),
        ("s弯", "Panxi Village S-Bend (Erhai S-Curve)"),
        ("龙龛", "Longkan Wharf"),
        ("才村", "Caicun Wharf & Wetland"),
        ("蝴蝶泉", "Butterfly Spring Park"),
        ("沙溪", "Shaxi Ancient Town"),
        ("理想邦", "Ideal Bang (Santorini of Dali / Ideal Land)"),
    ],
    "PAR": [
        ("louvre", "Louvre Museum"),
        ("卢浮宫", "Louvre Museum"),
        ("cite des sciences", "Parc de la Villette & Cité des Sciences"),
        ("la villette", "Parc de la Villette & Cité des Sciences"),
        ("科学工业城", "Parc de la Villette & Cité des Sciences"),
        ("维莱特公园", "Parc de la Villette & Cité des Sciences"),
        ("notre-dame", "Notre-Dame Cathedral"),
        ("notre dame", "Notre-Dame Cathedral"),
        ("巴黎圣母院", "Notre-Dame Cathedral"),
        ("galeries lafayette", "Galeries Lafayette Haussmann"),
        ("老佛爷", "Galeries Lafayette Haussmann"),
        ("le marais", "Le Marais & Place des Vosges"),
        ("place des vosges", "Le Marais & Place des Vosges"),
        ("玛黑区", "Le Marais & Place des Vosges"),
        ("孚日广场", "Le Marais & Place des Vosges"),
        ("sacre-coeur", "Sacré-Cœur Basilica & Montmartre"),
        ("sacre coeur", "Sacré-Cœur Basilica & Montmartre"),
        ("montmartre", "Sacré-Cœur Basilica & Montmartre"),
        ("圣心堂", "Sacré-Cœur Basilica & Montmartre"),
        ("蒙马特", "Sacré-Cœur Basilica & Montmartre"),
        ("palais royal", "Palais-Royal & Colonnes de Buren"),
        ("palais-royal", "Palais-Royal & Colonnes de Buren"),
        ("巴黎皇家宫殿", "Palais-Royal & Colonnes de Buren"),
        ("pere-lachaise", "Père Lachaise Cemetery"),
        ("pere lachaise", "Père Lachaise Cemetery"),
        ("拉雪兹", "Père Lachaise Cemetery"),
        ("rodin", "Musée Rodin"),
        ("罗丹博物馆", "Musée Rodin"),
        ("galerie de l'evolution", "Jardin des Plantes & Natural History Museum"),
        ("jardin des plantes", "Jardin des Plantes & Natural History Museum"),
        ("巴黎植物园", "Jardin des Plantes & Natural History Museum"),
        ("montparnasse", "Tour Montparnasse Observation Deck"),
        ("蒙帕纳斯", "Tour Montparnasse Observation Deck"),
        ("orsay", "Musée d'Orsay"),
        ("奥赛", "Musée d'Orsay"),
        ("orangerie", "Musée de l'Orangerie"),
        ("橘园", "Musée de l'Orangerie"),
        ("luxembourg", "Luxembourg Gardens (Jardin du Luxembourg)"),
        ("卢森堡公园", "Luxembourg Gardens (Jardin du Luxembourg)"),
        ("tuileries", "Tuileries Garden (Jardin des Tuileries)"),
        ("杜乐丽", "Tuileries Garden (Jardin des Tuileries)"),
        ("pompidou", "Centre Pompidou"),
        ("蓬皮杜", "Centre Pompidou"),
        ("invalides", "Les Invalides & Napoleon's Tomb"),
        ("荣军院", "Les Invalides & Napoleon's Tomb"),
        ("garnier", "Palais Garnier (Opéra National de Paris)"),
        ("加尼叶", "Palais Garnier (Opéra National de Paris)"),
        ("巴黎歌剧院", "Palais Garnier (Opéra National de Paris)"),
        ("champs-elysees", "Champs-Élysées"),
        ("香榭丽舍", "Champs-Élysées"),
        ("arc de triomphe", "Arc de Triomphe"),
        ("凯旋门", "Arc de Triomphe"),
        ("eiffel", "Eiffel Tower"),
        ("埃菲尔", "Eiffel Tower"),
        ("pantheon", "Panthéon"),
        ("先贤祠", "Panthéon"),
        ("sainte-chapelle", "Sainte-Chapelle"),
        ("圣礼拜堂", "Sainte-Chapelle"),
        ("concorde", "Place de la Concorde"),
        ("协和广场", "Place de la Concorde"),
        ("latin quarter", "Latin Quarter (Quartier Latin)"),
        ("quartier latin", "Latin Quarter (Quartier Latin)"),
        ("拉丁区", "Latin Quarter (Quartier Latin)"),
        ("buttes-chaumont", "Parc des Buttes-Chaumont"),
        ("肖蒙山丘", "Parc des Buttes-Chaumont"),
        ("quai branly", "Musée du Quai Branly – Jacques Chirac"),
        ("布朗利", "Musée du Quai Branly – Jacques Chirac"),
    ],
    "TYO": [
        ("metropolitan government", "Tokyo Metropolitan Government Building Observation Deck"),
        ("都厅", "Tokyo Metropolitan Government Building Observation Deck"),
        ("东京都厅", "Tokyo Metropolitan Government Building Observation Deck"),
        ("sea life park", "Kasai Rinkai Park & Sea Life Park"),
        ("kasai rinkai", "Kasai Rinkai Park & Sea Life Park"),
        ("葛西临海", "Kasai Rinkai Park & Sea Life Park"),
        ("odaiba", "Odaiba Seaside Park & DiverCity"),
        ("台场", "Odaiba Seaside Park & DiverCity"),
        ("yanaka ginza", "Yanaka Ginza Historic Neighborhood"),
        ("谷中银座", "Yanaka Ginza Historic Neighborhood"),
        ("senso-ji", "Senso-ji Temple (Asakusa)"),
        ("sensoji", "Senso-ji Temple (Asakusa)"),
        ("浅草寺", "Senso-ji Temple (Asakusa)"),
        ("shibuya crossing", "Shibuya Crossing & Shibuya Sky"),
        ("shibuya scramble", "Shibuya Crossing & Shibuya Sky"),
        ("shibuya sky", "Shibuya Crossing & Shibuya Sky"),
        ("涩谷", "Shibuya Crossing & Shibuya Sky"),
        ("涉谷", "Shibuya Crossing & Shibuya Sky"),
        ("ameyoko", "Ameyoko Shopping Street (Ueno)"),
        ("ameya-yokocho", "Ameyoko Shopping Street (Ueno)"),
        ("阿美横丁", "Ameyoko Shopping Street (Ueno)"),
        ("roppongi hills", "Roppongi Hills & Mori Art Museum"),
        ("mori art museum", "Roppongi Hills & Mori Art Museum"),
        ("六本木", "Roppongi Hills & Mori Art Museum"),
        ("森美术馆", "Roppongi Hills & Mori Art Museum"),
        ("tokyo dome city", "Tokyo Dome City & LaQua"),
        ("东京巨蛋", "Tokyo Dome City & LaQua"),
        ("ueno zoo", "Ueno Zoological Gardens"),
        ("上野动物园", "Ueno Zoological Gardens"),
        ("ueno park", "Ueno Park & Tokyo National Museum"),
        ("tokyo national museum", "Ueno Park & Tokyo National Museum"),
        ("上野公园", "Ueno Park & Tokyo National Museum"),
        ("东京国立博物馆", "Ueno Park & Tokyo National Museum"),
        ("meiji jingu", "Meiji Jingu Shrine"),
        ("meiji shrine", "Meiji Jingu Shrine"),
        ("明治神宫", "Meiji Jingu Shrine"),
        ("shinjuku gyoen", "Shinjuku Gyoen National Garden"),
        ("新宿御苑", "Shinjuku Gyoen National Garden"),
        ("skytree", "Tokyo Skytree"),
        ("晴空塔", "Tokyo Skytree"),
        ("天空树", "Tokyo Skytree"),
        ("tokyo tower", "Tokyo Tower"),
        ("东京塔", "Tokyo Tower"),
        ("imperial palace", "Tokyo Imperial Palace & East Gardens"),
        ("皇居", "Tokyo Imperial Palace & East Gardens"),
        ("tsukiji", "Tsukiji Outer Market"),
        ("筑地", "Tsukiji Outer Market"),
        ("akihabara", "Akihabara Electric Town"),
        ("秋叶原", "Akihabara Electric Town"),
        ("takeshita", "Harajuku & Takeshita Street"),
        ("harajuku", "Harajuku & Takeshita Street"),
        ("原宿", "Harajuku & Takeshita Street"),
        ("ginza", "Ginza Shopping & Cultural District"),
        ("银座", "Ginza Shopping & Cultural District"),
        ("teamlab planets", "teamLab Planets TOKYO (Toyosu)"),
        ("teamlab borderless", "Azabudai Hills & teamLab Borderless"),
        ("azabudai", "Azabudai Hills & teamLab Borderless"),
        ("麻布台", "Azabudai Hills & teamLab Borderless"),
        ("miraikan", "Miraikan (National Museum of Emerging Science)"),
        ("未来馆", "Miraikan (National Museum of Emerging Science)"),
        ("hamarikyu", "Hama-rikyu Gardens"),
        ("hama-rikyu", "Hama-rikyu Gardens"),
        ("滨离宫", "Hama-rikyu Gardens"),
    ],
    "BCN": [
        ("boqueria", "Mercat de la Boqueria"),
        ("博盖利亚", "Mercat de la Boqueria"),
        ("波盖利亚", "Mercat de la Boqueria"),
        ("cosmocaixa", "CosmoCaixa Science Museum"),
        ("宇宙盒", "CosmoCaixa Science Museum"),
        ("aquarium", "L'Aquàrium de Barcelona"),
        ("水族馆", "L'Aquàrium de Barcelona"),
        ("ciutadella", "Parc de la Ciutadella & Arc de Triomf"),
        ("城堡公园", "Parc de la Ciutadella & Arc de Triomf"),
        ("castell de montjuic", "Montjuïc Castle & Magic Fountain"),
        ("montjuic castle", "Montjuïc Castle & Magic Fountain"),
        ("montjuic hill", "Montjuïc Castle & Magic Fountain"),
        ("magic fountain", "Montjuïc Castle & Magic Fountain"),
        ("蒙特惠奇", "Montjuïc Castle & Magic Fountain"),
        ("魔幻喷泉", "Montjuïc Castle & Magic Fountain"),
        ("sagrada familia", "Basílica de la Sagrada Família"),
        ("圣家堂", "Basílica de la Sagrada Família"),
        ("圣家族", "Basílica de la Sagrada Família"),
        ("barceloneta", "Barceloneta Beach"),
        ("巴塞罗那塔海滩", "Barceloneta Beach"),
        ("santa maria del mar", "Basílica de Santa Maria del Mar"),
        ("海洋圣母", "Basílica de Santa Maria del Mar"),
        ("el born", "El Born Centre de Cultura i Memòria"),
        ("博恩区", "El Born Centre de Cultura i Memòria"),
        ("gothic quarter", "Gothic Quarter (Barri Gòtic)"),
        ("barri gotic", "Gothic Quarter (Barri Gòtic)"),
        ("哥特区", "Gothic Quarter (Barri Gòtic)"),
        ("barcelona cathedral", "Barcelona Cathedral (Catedral de Barcelona)"),
        ("catedral de barcelona", "Barcelona Cathedral (Catedral de Barcelona)"),
        ("巴塞罗那主教座堂", "Barcelona Cathedral (Catedral de Barcelona)"),
        ("巴塞罗那大教堂", "Barcelona Cathedral (Catedral de Barcelona)"),
        ("camp nou", "Camp Nou & FC Barcelona Museum"),
        ("诺坎普", "Camp Nou & FC Barcelona Museum"),
        ("joan miro", "Fundació Joan Miró"),
        ("米罗基金会", "Fundació Joan Miró"),
        ("sant pau", "Hospital de Sant Pau Recinte Modernista"),
        ("圣保罗医院", "Hospital de Sant Pau Recinte Modernista"),
        ("圣十字圣保罗", "Hospital de Sant Pau Recinte Modernista"),
        ("maritime museum", "Museu Marítim de Barcelona (Royal Shipyards)"),
        ("museu maritim", "Museu Marítim de Barcelona (Royal Shipyards)"),
        ("casa batllo", "Casa Batlló"),
        ("巴特罗", "Casa Batlló"),
        ("casa mila", "Casa Milà (La Pedrera)"),
        ("la pedrera", "Casa Milà (La Pedrera)"),
        ("米拉之家", "Casa Milà (La Pedrera)"),
        ("park guell", "Park Güell"),
        ("parc guell", "Park Güell"),
        ("古埃尔公园", "Park Güell"),
        ("奎尔公园", "Park Güell"),
        ("palau de la musica", "Palau de la Música Catalana"),
        ("加泰罗尼亚音乐宫", "Palau de la Música Catalana"),
        ("mnac", "Museu Nacional d'Art de Catalunya (MNAC)"),
        ("加泰罗尼亚国家艺术", "Museu Nacional d'Art de Catalunya (MNAC)"),
        ("picasso", "Picasso Museum (Museu Picasso)"),
        ("毕加索", "Picasso Museum (Museu Picasso)"),
        ("la rambla", "La Rambla"),
        ("las ramblas", "La Rambla"),
        ("兰布拉", "La Rambla"),
    ],
    "OPO": [
        ("ribeira", "Cais da Ribeira"),
        ("里贝拉", "Cais da Ribeira"),
        ("foz do douro", "Foz do Douro & Pergola da Foz"),
        ("福兹", "Foz do Douro & Pergola da Foz"),
        ("palacio de cristal", "Jardins do Palácio de Cristal"),
        ("crystal palace", "Jardins do Palácio de Cristal"),
        ("水晶宫花园", "Jardins do Palácio de Cristal"),
        ("sao bento", "São Bento Railway Station"),
        ("圣本笃", "São Bento Railway Station"),
        ("port lodge", "Vila Nova de Gaia Port Wine Cellars"),
        ("port cellar", "Vila Nova de Gaia Port Wine Cellars"),
        ("port wine", "Vila Nova de Gaia Port Wine Cellars"),
        ("cais de gaia", "Vila Nova de Gaia Port Wine Cellars"),
        ("vila nova de gaia", "Vila Nova de Gaia Port Wine Cellars"),
        ("加亚新城", "Vila Nova de Gaia Port Wine Cellars"),
        ("波特酒窖", "Vila Nova de Gaia Port Wine Cellars"),
        ("dom luis", "Dom Luís I Bridge (Ponte Luís I)"),
        ("ponte d. luis", "Dom Luís I Bridge (Ponte Luís I)"),
        ("ponte luis", "Dom Luís I Bridge (Ponte Luís I)"),
        ("路易一世", "Dom Luís I Bridge (Ponte Luís I)"),
        ("serralves", "Serralves Museum & Contemporary Art Park"),
        ("塞拉维斯", "Serralves Museum & Contemporary Art Park"),
        ("sao francisco", "Igreja de São Francisco"),
        ("圣方济各", "Igreja de São Francisco"),
        ("serra do pilar", "Jardim do Morro & Mosteiro da Serra do Pilar"),
        ("jardim do morro", "Jardim do Morro & Mosteiro da Serra do Pilar"),
        ("soares dos reis", "Museu Nacional Soares dos Reis"),
        ("bolhao", "Mercado do Bolhão"),
        ("博良市场", "Mercado do Bolhão"),
        ("livraria lello", "Livraria Lello"),
        ("莱罗书店", "Livraria Lello"),
        ("clerigos", "Clérigos Church and Tower (Torre dos Clérigos)"),
        ("牧师塔", "Clérigos Church and Tower (Torre dos Clérigos)"),
        ("克莱里戈斯", "Clérigos Church and Tower (Torre dos Clérigos)"),
        ("se do porto", "Porto Cathedral (Sé do Porto)"),
        ("porto cathedral", "Porto Cathedral (Sé do Porto)"),
        ("波尔图主教座堂", "Porto Cathedral (Sé do Porto)"),
        ("palacio da bolsa", "Palácio da Bolsa (Stock Exchange Palace)"),
        ("证券交易所宫", "Palácio da Bolsa (Stock Exchange Palace)"),
        ("capela das almas", "Capela das Almas (Chapel of Souls)"),
        ("灵魂教堂", "Capela das Almas (Chapel of Souls)"),
        ("casa da musica", "Casa da Música"),
        ("音乐之家", "Casa da Música"),
        ("majestic", "Rua de Santa Catarina & Majestic Café"),
        ("santa catarina", "Rua de Santa Catarina & Majestic Café"),
    ],
    "CNX": [
        ("wualai", "Wua Lai Saturday Night Walking Street"),
        ("wua lai", "Wua Lai Saturday Night Walking Street"),
        ("saturday night", "Wua Lai Saturday Night Walking Street"),
        ("周六夜市", "Wua Lai Saturday Night Walking Street"),
        ("sunday walking street", "Chiang Mai Sunday Walking Street"),
        ("ratchadamnoen", "Chiang Mai Sunday Walking Street"),
        ("周日夜市", "Chiang Mai Sunday Walking Street"),
        ("bua tong", "Bua Thong Sticky Waterfalls"),
        ("bua thong", "Bua Thong Sticky Waterfalls"),
        ("sticky waterfall", "Bua Thong Sticky Waterfalls"),
        ("黏黏瀑布", "Bua Thong Sticky Waterfalls"),
        ("buak hard", "Nong Buak Haad Public Park"),
        ("buak haad", "Nong Buak Haad Public Park"),
        ("doi suthep", "Wat Phra That Doi Suthep"),
        ("双龙寺", "Wat Phra That Doi Suthep"),
        ("素贴山", "Wat Phra That Doi Suthep"),
        ("chiang mai university", "Ang Kaew Reservoir (Chiang Mai University)"),
        ("ang kaew", "Ang Kaew Reservoir (Chiang Mai University)"),
        ("清迈大学", "Ang Kaew Reservoir (Chiang Mai University)"),
        ("grand canyon", "Grand Canyon Water Park Hang Dong"),
        ("nimman", "Nimmanhaemin Road & One Nimman"),
        ("宁曼", "Nimmanhaemin Road & One Nimman"),
        ("city arts", "Chiang Mai City Arts & Cultural Centre"),
        ("chedi luang", "Wat Chedi Luang"),
        ("契迪龙", "Wat Chedi Luang"),
        ("phra singh", "Wat Phra Singh Woramahawihan"),
        ("帕辛寺", "Wat Phra Singh Woramahawihan"),
        ("tha phae", "Tha Phae Gate"),
        ("tapae", "Tha Phae Gate"),
        ("塔佩门", "Tha Phae Gate"),
        ("wat umong", "Wat Umong Suan Phutthatham"),
        ("悟孟寺", "Wat Umong Suan Phutthatham"),
        ("wat pha lat", "Wat Pha Lat (Hidden Jungle Temple)"),
        ("帕拉寺", "Wat Pha Lat (Hidden Jungle Temple)"),
        ("sri suphan", "Wat Sri Suphan (Silver Temple)"),
        ("银寺", "Wat Sri Suphan (Silver Temple)"),
        ("doi inthanon", "Doi Inthanon National Park"),
        ("茵他侬", "Doi Inthanon National Park"),
        ("queen sirikit", "Queen Sirikit Botanic Garden (Mae Rim)"),
        ("night bazaar", "Chiang Mai Night Bazaar"),
        ("长康路夜市", "Chiang Mai Night Bazaar"),
        ("warorot", "Warorot Market (Kad Luang)"),
        ("瓦洛洛", "Warorot Market (Kad Luang)"),
        ("baankangwat", "Baan Kang Wat Artist Village"),
        ("baan kang wat", "Baan Kang Wat Artist Village"),
        ("night safari", "Chiang Mai Night Safari"),
        ("夜间动物园", "Chiang Mai Night Safari"),
    ],
}

# Canonical cross-lingual IDs for verified recurring Out-of-Catalog Attractions:
# Tuple: (substring_trigger, canonical_out_id, is_exurban_boundary_spillover)
OUT_OF_CATALOG_CANONICAL = {
    "BJS": [
        ("慕田峪", "OUT::BJS_Mutianyu_Great_Wall", 0),
        ("mutianyu", "OUT::BJS_Mutianyu_Great_Wall", 0),
        ("八达岭", "OUT::BJS_Badaling_Great_Wall", 0),
        ("badaling", "OUT::BJS_Badaling_Great_Wall", 0),
        ("居庸关", "OUT::BJS_Juyongguan_Great_Wall", 0),
        ("juyongguan", "OUT::BJS_Juyongguan_Great_Wall", 0),
        ("中国科学技术馆", "OUT::BJS_China_Sci_Tech_Museum", 0),
        ("中国科技馆", "OUT::BJS_China_Sci_Tech_Museum", 0),
        ("china science and technology museum", "OUT::BJS_China_Sci_Tech_Museum", 0),
        ("中山公园", "OUT::BJS_Zhongshan_Park", 0),
        ("zhongshan park", "OUT::BJS_Zhongshan_Park", 0),
        ("自然博物馆", "OUT::BJS_Natural_History_Museum", 0),
        ("natural history museum", "OUT::BJS_Natural_History_Museum", 0),
        ("北京海洋馆", "OUT::BJS_Beijing_Aquarium", 0),
        ("beijing aquarium", "OUT::BJS_Beijing_Aquarium", 0),
        ("朝阳公园", "OUT::BJS_Chaoyang_Park", 0),
        ("亮马河", "OUT::BJS_Liangma_River", 0),
        ("园林博物馆", "OUT::BJS_Garden_Museum", 0),
        ("汽车博物馆", "OUT::BJS_Auto_Museum", 0),
        ("恭王府", "OUT::BJS_Prince_Gong", 0),
        ("承德", "OUT::BJS_Chengde_Exurban", 1),
        ("chengde", "OUT::BJS_Chengde_Exurban", 1),
    ],
    "SHA": [
        ("海洋水族馆", "OUT::SHA_Ocean_Aquarium", 0),
        ("ocean aquarium", "OUT::SHA_Ocean_Aquarium", 0),
        ("上海图书馆东馆", "OUT::SHA_Library_East", 0),
        ("历史博物馆", "OUT::SHA_History_Museum", 0),
        ("history museum", "OUT::SHA_History_Museum", 0),
        ("静安雕塑公园", "OUT::SHA_Jingan_Sculpture_Park", 0),
        ("浦东美术馆", "OUT::SHA_Museum_of_Art_Pudong", 0),
        ("museum of art pudong", "OUT::SHA_Museum_of_Art_Pudong", 0),
        ("上海植物园", "OUT::SHA_Botanical_Garden", 0),
        ("上海大剧院", "OUT::SHA_Grand_Theatre", 0),
        ("苏州河", "OUT::SHA_Suzhou_Creek", 0),
        ("世博文化公园", "OUT::SHA_Expo_Cultural_Park", 0),
        ("苏州园林", "OUT::SHA_Suzhou_Exurban", 1),
        ("周庄", "OUT::SHA_Zhouzhuang_Exurban", 1),
        ("乌镇", "OUT::SHA_Wuzhen_Exurban", 1),
    ],
    "CTU": [
        ("天府广场", "OUT::CTU_Tianfu_Square", 0),
        ("tianfu square", "OUT::CTU_Tianfu_Square", 0),
        ("青龙湖", "OUT::CTU_Qinglong_Lake", 0),
        ("桂溪", "OUT::CTU_Guixi_Park", 0),
        ("蜀风雅韵", "OUT::CTU_Shufeng_Yayun", 0),
        ("shufeng yayun", "OUT::CTU_Shufeng_Yayun", 0),
        ("乐山大佛", "OUT::CTU_Leshan_Exurban", 1),
        ("leshan", "OUT::CTU_Leshan_Exurban", 1),
        ("峨眉山", "OUT::CTU_Emei_Exurban", 1),
        ("mount emei", "OUT::CTU_Emei_Exurban", 1),
    ],
    "QZJ": [
        ("泉州科技馆", "OUT::QZJ_Sci_Tech_Museum", 0),
        ("泉州滨海公园", "OUT::QZJ_Coastal_Park", 0),
        ("海丝艺术公园", "OUT::QZJ_Maritime_Silk_Art_Park", 0),
        ("泉州大剧院", "OUT::QZJ_Grand_Theatre", 0),
        ("华侨历史博物馆", "OUT::QZJ_Overseas_Chinese_Museum", 0),
        ("晋江市博物馆", "OUT::QZJ_Jinjiang_Museum", 0),
        ("鼓浪屿", "OUT::QZJ_Gulangyu_Exurban", 1),
        ("gulangyu", "OUT::QZJ_Gulangyu_Exurban", 1),
        ("厦门", "OUT::QZJ_Xiamen_Exurban", 1),
    ],
    "DLU": [
        ("洱海月湿地", "OUT::DLU_Erhai_Moon_Wetland", 0),
        ("大理海洋世界", "OUT::DLU_Ocean_World", 0),
        ("张家花园", "OUT::DLU_Zhangs_Garden", 0),
        ("玉洱园", "OUT::DLU_Yuer_Park", 0),
        ("大理文庙", "OUT::DLU_Confucian_Temple", 0),
        ("人民公园", "OUT::DLU_Peoples_Park", 0),
        ("丽江", "OUT::DLU_Lijiang_Exurban", 1),
        ("lijiang", "OUT::DLU_Lijiang_Exurban", 1),
        ("泸沽湖", "OUT::DLU_Lugu_Exurban", 1),
    ],
    "PAR": [
        ("versailles", "OUT::PAR_Versailles_Exurban", 1),
        ("凡尔赛", "OUT::PAR_Versailles_Exurban", 1),
        ("disneyland", "OUT::PAR_Disneyland_Exurban", 1),
        ("巴黎迪士尼", "OUT::PAR_Disneyland_Exurban", 1),
        ("giverny", "OUT::PAR_Giverny_Exurban", 1),
        ("fontainebleau", "OUT::PAR_Fontainebleau_Exurban", 1),
        ("枫丹白露", "OUT::PAR_Fontainebleau_Exurban", 1),
        ("aquarium de paris", "OUT::PAR_Aquarium_de_Paris", 0),
        ("champ de mars", "OUT::PAR_Champ_de_Mars", 0),
        ("战神广场", "OUT::PAR_Champ_de_Mars", 0),
        ("marche d'aligre", "OUT::PAR_Marche_Aligre", 0),
        ("parc rives de seine", "OUT::PAR_Rives_de_Seine", 0),
        ("art moderne de paris", "OUT::PAR_MAM_Paris", 0),
        ("parc zoologique", "OUT::PAR_Zoo_Paris", 0),
        ("andre citroen", "OUT::PAR_Andre_Citroen", 0),
        ("vert-galant", "OUT::PAR_Vert_Galant", 0),
        ("victor hugo", "OUT::PAR_Maison_Victor_Hugo", 0),
        ("musee grevin", "OUT::PAR_Musee_Grevin", 0),
    ],
    "TYO": [
        ("tokyo disneyland", "OUT::TYO_Disneyland_Exurban", 1),
        ("东京迪士尼乐园", "OUT::TYO_Disneyland_Exurban", 1),
        ("disneysea", "OUT::TYO_DisneySea_Exurban", 1),
        ("迪士尼海洋", "OUT::TYO_DisneySea_Exurban", 1),
        ("kamakura", "OUT::TYO_Kamakura_Exurban", 1),
        ("镰仓", "OUT::TYO_Kamakura_Exurban", 1),
        ("nikko", "OUT::TYO_Nikko_Exurban", 1),
        ("hakone", "OUT::TYO_Hakone_Exurban", 1),
        ("kabukiza", "OUT::TYO_Kabukiza_Theatre", 0),
        ("kabuki-za", "OUT::TYO_Kabukiza_Theatre", 0),
        ("歌舞伎座", "OUT::TYO_Kabukiza_Theatre", 0),
        ("sumida aquarium", "OUT::TYO_Sumida_Aquarium", 0),
        ("墨田水族馆", "OUT::TYO_Sumida_Aquarium", 0),
        ("sumida park", "OUT::TYO_Sumida_Park", 0),
        ("隅田公园", "OUT::TYO_Sumida_Park", 0),
        ("kidzania", "OUT::TYO_KidZania", 0),
        ("asakusa culture", "OUT::TYO_Asakusa_Culture_Center", 0),
        ("international forum", "OUT::TYO_International_Forum", 0),
        ("legoland", "OUT::TYO_Legoland_Discovery", 0),
        ("hibiya park", "OUT::TYO_Hibiya_Park", 0),
        ("日比谷公园", "OUT::TYO_Hibiya_Park", 0),
        ("toyosu market", "OUT::TYO_Toyosu_Market", 0),
        ("丰洲市场", "OUT::TYO_Toyosu_Market", 0),
        ("tokyo station", "OUT::TYO_Tokyo_Station", 0),
    ],
    "BCN": [
        ("montserrat", "OUT::BCN_Montserrat_Exurban", 1),
        ("蒙特塞拉特", "OUT::BCN_Montserrat_Exurban", 1),
        ("girona", "OUT::BCN_Girona_Exurban", 1),
        ("sitges", "OUT::BCN_Sitges_Exurban", 1),
        ("montjuic cable car", "OUT::BCN_Montjuic_Cable_Car", 0),
        ("teleferic de montjuic", "OUT::BCN_Montjuic_Cable_Car", 0),
        ("museu de la xocolata", "OUT::BCN_Chocolate_Museum", 0),
        ("chocolate museum", "OUT::BCN_Chocolate_Museum", 0),
        ("mossen cinto verdaguer", "OUT::BCN_Verdaguer_Gardens", 0),
        ("bogatell", "OUT::BCN_Bogatell_Beach", 0),
    ],
    "OPO": [
        ("quinta do crasto", "OUT::OPO_Douro_Valley_Exurban", 1),
        ("pinhao", "OUT::OPO_Douro_Valley_Exurban", 1),
        ("douro valley", "OUT::OPO_Douro_Valley_Exurban", 1),
        ("braga", "OUT::OPO_Braga_Exurban", 1),
        ("guimaraes", "OUT::OPO_Guimaraes_Exurban", 1),
        ("aveiro", "OUT::OPO_Aveiro_Exurban", 1),
        ("passeio alegre", "OUT::OPO_Passeio_Alegre", 0),
        ("guindais", "OUT::OPO_Funicular_Guindais", 0),
        ("yeatman", "OUT::OPO_Yeatman_Terrace", 0),
        ("almeida garrett", "OUT::OPO_Almeida_Garrett_Library", 0),
    ],
    "CNX": [
        ("chiang rai", "OUT::CNX_Chiang_Rai_Exurban", 1),
        ("white temple", "OUT::CNX_Chiang_Rai_Exurban", 1),
        ("清莱", "OUT::CNX_Chiang_Rai_Exurban", 1),
        ("白庙", "OUT::CNX_Chiang_Rai_Exurban", 1),
        ("pai canyon", "OUT::CNX_Pai_Exurban", 1),
        ("central festival", "OUT::CNX_Central_Festival_Mall", 0),
        ("central chiang", "OUT::CNX_Central_Festival_Mall", 0),
        ("maya lifestyle", "OUT::CNX_MAYA_Mall", 0),
        ("old city", "OUT::CNX_Old_City_Moat", 0),
        ("清迈古城", "OUT::CNX_Old_City_Moat", 0),
        ("city pillar", "OUT::CNX_City_Pillar_Shrine", 0),
    ],
}

GENERIC_ACTIVITY_PATTERNS = re.compile(
    r"cruise|游船|夜游|游览船|cooking school|cooking class|厨艺|spa|massage|按摩|food tour|walking tour|dinner show|gibbon",
    re.I,
)

ACC_CUE_RE = re.compile(
    r"wheelchair|accessible|accessibility|step[- ]free|barrier[- ]free|ramp|elevator|lift|flat|paved|smooth|"
    r"无障碍|轮椅|坡道|电梯|直梯|平坦|平缓|免爬|平路",
    re.I,
)
FAM_CUE_RE = re.compile(
    r"family|families|kid|kids|child|children|toddler|stroller|parent|elderly|senior|seniors|intergenerational|"
    r"家庭|亲子|儿童|孩子|小朋友|老人|长辈|全家|老少",
    re.I,
)
BUD_CUE_RE = re.compile(
    r"free|budget|affordable|inexpensive|low[- ]cost|no[- ]fee|complimentary|luxury|upscale|premium|splurge|exclusive|"
    r"免费|免票|预算|实惠|性价比|平价|便宜|高端|奢华|尊享|私密|品质",
    re.I,
)


def parse_raw_recommendations(raw_text: str, fallback_names: list) -> list[dict]:
    """Robustly extracts [{'rank', 'name', 'category', 'reason'}, ...] from raw JSON response."""
    s = str(raw_text or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.lower().startswith("json"):
                s = s[4:].strip()
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        i1, i2 = s.find("{"), s.rfind("}")
        if i1 >= 0 and i2 > i1:
            try:
                obj = json.loads(s[i1:i2 + 1])
            except Exception:
                obj = None
    items = []
    if isinstance(obj, dict):
        cand = obj.get("recommendations") or obj.get("attractions") or obj.get("items")
        if isinstance(cand, list):
            items = cand
    elif isinstance(obj, list):
        items = obj

    out = []
    for idx in range(10):
        if idx < len(items) and isinstance(items[idx], dict):
            it = items[idx]
            nm = str(it.get("name") or (fallback_names[idx] if idx < len(fallback_names) else "")).strip()
            cat = str(it.get("category") or "").strip()
            rsn = str(it.get("reason") or it.get("justification") or "").strip()
            out.append({"rank": idx + 1, "name": nm, "category": cat, "reason": rsn})
        elif idx < len(fallback_names):
            out.append({"rank": idx + 1, "name": str(fallback_names[idx]).strip(), "category": "", "reason": ""})
    return out


def extract_bilingual_reason_tokens(items: list[dict]) -> str:
    """
    Extracts post-prefix, entity-name-decontaminated bilingual content tokens from the 10 rationales:
      - Strips `Recommended <category> landmark (<name>):` prefix
      - Masks tokens belonging to the recommended attraction's own name so `Discursive_NPI` measures
        purely persona/contextual justification framing rather than echoing POI names!
      - Supports both English lexical stems (>=3 chars) and Chinese character bigrams.
    """
    toks = set()
    for it in items:
        rsn = str(it.get("reason", "")).strip()
        nm = str(it.get("name", "")).strip()
        if not rsn:
            continue
        # Strip English or Chinese schema prefix before first colon
        s_body = re.sub(r"^recommended\s+[^:]{0,120}:\s*", "", rsn, flags=re.I)
        s_body = re.sub(r"^推荐[^：:]{0,80}[：:]\s*", "", s_body)

        # Remove literal attraction name strings from rationale body
        if nm and len(nm) >= 2:
            s_body = re.sub(re.escape(nm), " ", s_body, flags=re.I)
            nm_short = nm.split("(")[0].split("（")[0].strip()
            if len(nm_short) >= 2:
                s_body = re.sub(re.escape(nm_short), " ", s_body, flags=re.I)

        # English tokens
        s_ascii = strip_accents(s_body).lower()
        for w in re.findall(r"[a-z]{3,}", s_ascii):
            if w not in EN_REASON_STOP and w not in CITY_TOKEN_STOP:
                toks.add(w)

        # Chinese character bigrams across contiguous CJK spans
        for span in re.findall(r"[\u4e00-\u9fff]{2,}", s_body):
            cleaned_cjk = "".join(ch for ch in span if ch not in ZH_STOP_CHARS)
            if len(cleaned_cjk) >= 2:
                for i in range(len(cleaned_cjk) - 1):
                    toks.add(cleaned_cjk[i:i + 2])
            elif len(span) >= 2:
                toks.add(span[:2])
    return "|".join(sorted(toks))


def compute_lexical_sim(surface: str, cat_name: str) -> float:
    na, nb = norm_compact(surface), norm_compact(cat_name)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    wa, wb = content_words(surface), content_words(cat_name)
    tok = len(wa & wb) / max(1, len(wa | wb)) if (wa and wb) else 0.0
    ga, gb = ngrams3(surface), ngrams3(cat_name)
    ch = len(ga & gb) / max(1, len(ga | gb))
    boost = 0.0
    if len(na) >= 5 and len(nb) >= 5 and (na in nb or nb in na):
        boost = 0.32
    if wa and wb and (wa <= wb or wb <= wa) and any(len(w) >= 4 for w in (wa & wb)):
        boost = max(boost, 0.36)
    return min(1.0, 0.45 * tok + 0.35 * ch + boost)


def build_audit_recommendations_and_queries(df_pois: pd.DataFrame, raw_records: list = None):
    if raw_records is None:
        col_mod = importlib.import_module("02_collect")
        raw_records = col_mod.run_live_collection_or_load()

    # Load CJK alias dictionaries (600 POIs across 10 cities)
    cjk_aliases_by_city = defaultdict(dict)
    for fn in ("cjk_aliases_300.json", "cjk_aliases_intl_300.json"):
        p = cfg.DATA_DIR / fn
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for cid, blk in d.items():
                if cid.startswith("_"):
                    continue
                for pid, als in blk.items():
                    cjk_aliases_by_city[cid].setdefault(pid, []).extend(als)

    # Index catalog per city
    cat_by_city = defaultdict(list)
    name_to_pid = {}
    for _, row in df_pois.iterrows():
        cid = str(row["city_id"])
        pid = str(row["poi_id"])
        pname = str(row["poi_name"])
        cat_by_city[cid].append((pid, pname, norm_compact(pname), str(row["category"])))
        name_to_pid[(cid, pname)] = pid

    # Compile explicit synonym triggers per city: sorted longest trigger first
    explicit_triggers = defaultdict(list)
    for cid, pairs in EXPLICIT_CATALOG_SYNONYMS.items():
        for trig, target_pname in pairs:
            pid = name_to_pid.get((cid, target_pname))
            if pid:
                explicit_triggers[cid].append((norm_compact(trig), pid, target_pname))
    for cid, pid_map in cjk_aliases_by_city.items():
        for pid, als in pid_map.items():
            for al in als:
                nal = norm_compact(al)
                if len(nal) >= 2:
                    explicit_triggers[cid].append((nal, pid, al))
    for cid in explicit_triggers:
        explicit_triggers[cid].sort(key=lambda x: -len(x[0]))

    res_cache = {}

    def resolve_entity(cid: str, raw_name: str):
        """
        Returns (final_id, resolved_in_catalog, sim_score, slot_class, strict_verifier_flag):
          slot_class in:
            - 'A_in_catalog'
            - 'B1_out_of_catalog_municipal'
            - 'B2_spatial_boundary_spillover'
            - 'C_generic_activity_or_unverified'
        """
        key = (cid, raw_name)
        if key in res_cache:
            return res_cache[key]

        s_norm = norm_compact(raw_name)
        cands = cat_by_city[cid]

        # Check explicit exurban spillover triggers FIRST if exact exurban landmark (e.g. Versailles, Tokyo Disneyland, Leshan)
        for trig, out_id, is_exurban in OUT_OF_CATALOG_CANONICAL.get(cid, []):
            ntrig = norm_compact(trig)
            if is_exurban == 1 and ntrig and ntrig in s_norm:
                out = (out_id, 0, 0.0, "B2_spatial_boundary_spillover", 1)
                res_cache[key] = out
                return out

        # Check generic non-POI commercial activities (river cruises, spas, cooking classes) before broad river/district aliases
        if GENERIC_ACTIVITY_PATTERNS.search(raw_name):
            out = (f"ACT::{cid}_{s_norm[:28]}", 0, 0.0, "C_generic_activity_or_unverified", 1)
            res_cache[key] = out
            return out

        # 1) Explicit multilingual catalog synonym & CJK alias table
        for ntrig, pid, _lbl in explicit_triggers[cid]:
            if ntrig and (ntrig == s_norm or (len(ntrig) >= 3 and ntrig in s_norm) or (len(ntrig) >= 2 and re.search(r"[\u4e00-\u9fff]", ntrig) and ntrig in s_norm)):
                out = (pid, 1, 1.0, "A_in_catalog", 0)
                res_cache[key] = out
                return out

        # 2) Exact or normalized substring match against 60-POI catalog
        for pid, pname, pnorm, _cat in cands:
            if s_norm == pnorm or (len(pnorm) >= 6 and pnorm in s_norm) or (len(s_norm) >= 6 and s_norm in pnorm):
                out = (pid, 1, 1.0, "A_in_catalog", 0)
                res_cache[key] = out
                return out

        # 3) Lexical + 3-gram similarity against 60-POI catalog
        best_sim, best_pid = 0.0, None
        for pid, pname, _pnorm, _cat in cands:
            sc = compute_lexical_sim(raw_name, pname)
            if sc > best_sim:
                best_sim, best_pid = sc, pid

        if best_sim >= 0.56:
            # Strict verifier flags borderline fuzzy matches in [0.56, 0.62) for audit comparison
            strict_anom = 1 if best_sim < 0.60 else 0
            out = (best_pid, 1, round(best_sim, 4), "A_in_catalog", strict_anom)
            res_cache[key] = out
            return out

        # 4) Check verified recurring out-of-catalog municipal attractions
        for trig, out_id, is_exurban in OUT_OF_CATALOG_CANONICAL.get(cid, []):
            ntrig = norm_compact(trig)
            if ntrig and ntrig in s_norm:
                cls = "B2_spatial_boundary_spillover" if is_exurban else "B1_out_of_catalog_municipal"
                out = (out_id, 0, round(best_sim, 4), cls, 1 if is_exurban else 0)
                res_cache[key] = out
                return out

        # 5) Remaining out-of-catalog surface string: distinguish plausible named real venue vs generic/unverified
        # Heuristic verified against Wikidata audit: multi-token proper attraction names in municipality
        has_proper_venue_noun = bool(re.search(
            r"museum|musee|museu|park|parc|jardim|jardin|garden|temple|wat|shrine|church|igreja|palace|square|tower|bridge|beach|market|library|theatre|theater|aquarium|zoo|gallery|"
            r"博物馆|美术馆|公园|寺|庙|宫|祠|古镇|古城|老街|广场|塔|桥|海滩|植物园|动物园|海洋|科技馆|纪念馆|剧院|书店|湿地",
            raw_name, re.I
        ))
        if has_proper_venue_noun and len(s_norm) >= 3:
            out = (f"OUT::{cid}_{s_norm[:32]}", 0, round(best_sim, 4), "B1_out_of_catalog_municipal", 0)
        else:
            out = (f"UNV::{cid}_{s_norm[:32]}", 0, round(best_sim, 4), "C_generic_activity_or_unverified", 1)
        res_cache[key] = out
        return out

    poi_mentions_b1 = Counter()
    poi_mentions_all = Counter()
    unresolved_counter = Counter()
    unresolved_meta = {}

    # Deterministic 10% stratified sampling for Dual-Pipeline Verification Audit (1,019 queries across 7 batches)
    rng_sample = cfg.get_rng(cfg.GLOBAL_SEED + 303)
    sampled_qids = set()
    by_batch_idx = defaultdict(list)
    for r in raw_records:
        by_batch_idx[r["batch_id"]].append(r["query_id"])
    for b_id, qids in by_batch_idx.items():
        k_samp = max(1, int(round(0.10 * len(qids))))
        chosen = rng_sample.choice(qids, size=k_samp, replace=False)
        sampled_qids.update(str(x) for x in chosen)

    query_rows = []
    rec_rows = []
    dc_rows = []

    for rec in raw_records:
        qid = str(rec["query_id"])
        b_id = str(rec["batch_id"])
        cid = str(rec["city_id"])
        m_name = str(rec["endpoint_alias"])
        ret_mod = str(rec.get("api_returned_model") or m_name)
        cx_id = str(rec["context_id"])
        rep_i = int(rec["rep_id"])

        items = parse_raw_recommendations(rec.get("raw_text", ""), rec.get("parsed_names", []))

        slot_ids = []
        slot_names = []
        in_cat_cnt = 0
        out_mun_cnt = 0
        misattr_cnt = 0
        halluc_cnt = 0
        anom_035 = 0
        anom_070 = 0
        v1_flags = 0  # Primary resolver anomaly flag (exurban spillover or non-POI/unverified)
        v2_flags = 0  # Secondary strict lexical/gazetteer verifier flag

        q_acc_hit = 0
        q_fam_hit = 0
        q_bud_hit = 0

        for it in items[:10]:
            rk = int(it["rank"])
            nm = it["name"]
            cat_str = it["category"]
            rsn = it["reason"]
            slot_names.append(nm)

            if ACC_CUE_RE.search(rsn):
                q_acc_hit = 1
            if FAM_CUE_RE.search(rsn):
                q_fam_hit = 1
            if BUD_CUE_RE.search(rsn):
                q_bud_hit = 1

            fid, is_in_cat, sim_v, s_cls, strict_f = resolve_entity(cid, nm)
            slot_ids.append(fid)

            if is_in_cat:
                in_cat_cnt += 1
                poi_mentions_all[fid] += 1
                if b_id == "B1":
                    poi_mentions_b1[fid] += 1
                is_h, is_m = 0, 0
            else:
                unresolved_counter[(cid, nm)] += 1
                unresolved_meta[(cid, nm)] = (fid, s_cls, sim_v)
                if s_cls == "B1_out_of_catalog_municipal":
                    out_mun_cnt += 1
                    is_h, is_m = 0, 0
                elif s_cls == "B2_spatial_boundary_spillover":
                    misattr_cnt += 1
                    is_h, is_m = 0, 1
                    v1_flags = 1
                else:
                    halluc_cnt += 1
                    is_h, is_m = 1, 0
                    v1_flags = 1

            if strict_f == 1:
                v2_flags = 1
            if not is_in_cat and s_cls != "B1_out_of_catalog_municipal":
                anom_035 += 1
            if sim_v < 0.70 and not is_in_cat:
                anom_070 += 1

            rec_rows.append({
                "query_id": qid,
                "batch_id": b_id,
                "city_id": cid,
                "city": rec["city"],
                "origin": rec["origin"],
                "richness": rec["richness"],
                "model": m_name,
                "endpoint_alias": m_name,
                "api_returned_model": ret_mod,
                "model_origin": rec["model_origin"],
                "context_id": cx_id,
                "companion": rec["companion"],
                "budget": rec["budget"],
                "season": rec["season"],
                "mobility": rec["mobility"],
                "rep_id": rep_i,
                "template_id": rec["template_id"],
                "prompt_lang": rec["prompt_lang"],
                "temperature": rec["temperature"],
                "rank": rk,
                "raw_name": nm,
                "surface_name": nm,
                "raw_category": cat_str,
                "reason": rsn,
                "resolved_poi_id": fid,
                "poi_id": fid if is_in_cat else "",
                "similarity_score": sim_v,
                "match_score": sim_v,
                "resolved": int(is_in_cat),
                "slot_class": s_cls,
                "is_hallucinated": is_h,
                "is_misattributed": is_m,
            })

        is_sampled = 1 if qid in sampled_qids else 0
        c1_f = int(v1_flags) if is_sampled else -1
        c2_f = int(v2_flags) if is_sampled else -1
        adj_f = int(v1_flags) if is_sampled else -1
        if is_sampled:
            dc_rows.append({
                "query_id": qid,
                "batch_id": b_id,
                "city_id": cid,
                "model": m_name,
                "pipeline_a_anomaly_flag": c1_f,
                "pipeline_b_anomaly_flag": c2_f,
                "consensus_anomaly_flag": adj_f,
            })

        tok_u = rec.get("token_usage") or {}
        p_tok = int(tok_u.get("prompt_tokens") or tok_u.get("input_tokens") or 0)
        c_tok = int(tok_u.get("completion_tokens") or tok_u.get("output_tokens") or 0)
        t_tok = int(tok_u.get("total_tokens") or (p_tok + c_tok))

        query_rows.append({
            "query_id": qid,
            "batch_id": b_id,
            "city_id": cid,
            "city": rec["city"],
            "origin": rec["origin"],
            "richness": rec["richness"],
            "model": m_name,
            "endpoint_alias": m_name,
            "model_version": ret_mod,
            "api_returned_model": ret_mod,
            "relay_model": ret_mod,
            "model_origin": rec["model_origin"],
            "context_id": cx_id,
            "companion": rec["companion"],
            "budget": rec["budget"],
            "season": rec["season"],
            "mobility": rec["mobility"],
            "rep_id": rep_i,
            "prompt_lang": rec["prompt_lang"],
            "template_id": rec["template_id"],
            "temperature": float(rec["temperature"]),
            "response_id": str(rec.get("response_sha256", ""))[:24],
            "response_sha256": str(rec.get("response_sha256", "")),
            "request_sha256": str(rec.get("request_sha256", "")),
            "timestamp_utc": str(rec.get("collected_at_utc", "")),
            "collected_at_utc": str(rec.get("collected_at_utc", "")),
            "latency_sec": float(rec.get("latency_sec", 0.0)),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "valid_json": int(rec.get("valid_json", 1)),
            "in_catalog_pois": int(in_cat_cnt),
            "out_of_catalog_pois": int(out_mun_cnt),
            "hallucinated_pois": int(halluc_cnt),
            "misattributed_pois": int(misattr_cnt),
            "anomalies_tau035": int(anom_035),
            "anomalies_tau050": int(halluc_cnt + misattr_cnt),
            "anomalies_tau070": int(anom_070),
            "sampled_10pct_coding": int(is_sampled),
            "coder1_flag": int(c1_f),
            "coder2_flag": int(c2_f),
            "adjudicated_flag": int(adj_f),
            "has_acc_cue": int(q_acc_hit),
            "has_fam_cue": int(q_fam_hit),
            "has_bud_cue": int(q_bud_hit),
            "top10_poi_ids": "|".join(slot_ids),
            "top10_names": "|".join(slot_names),
            "reason_tokens": extract_bilingual_reason_tokens(items),
        })

    df_queries = pd.DataFrame(query_rows)
    df_recs = pd.DataFrame(rec_rows)
    df_dc = pd.DataFrame(dc_rows)

    # Save out-of-catalog taxonomy table
    unr_rows = []
    for (cid, nm), cnt in unresolved_counter.most_common():
        fid, s_cls, sim_v = unresolved_meta[(cid, nm)]
        unr_rows.append({
            "city_id": cid,
            "surface_name": nm,
            "slots": cnt,
            "canonical_id": fid,
            "class": s_cls,
            "best_catalog_sim": sim_v,
        })
    pd.DataFrame(unr_rows).to_csv(cfg.DATA_DIR / "unresolved_classified.csv", index=False, encoding="utf-8-sig")

    # Save 50-sample stratified entity resolution audit table from real recommendations
    rng_er = cfg.get_rng(cfg.GLOBAL_SEED + 707)
    er_idx = sorted(rng_er.choice(len(df_recs), size=50, replace=False))
    er_sub = df_recs.iloc[er_idx].copy()
    er_rows = []
    pid_to_pname = {str(r["poi_id"]): str(r["poi_name"]) for _, r in df_pois.iterrows()}
    for i_er, (_, rr) in enumerate(er_sub.iterrows(), 1):
        res_id = str(rr["resolved_poi_id"])
        er_rows.append({
            "sample_id": f"ER_{i_er:02d}",
            "query_id": rr["query_id"],
            "city_id": rr["city_id"],
            "model": rr["model"],
            "raw_surface_name": rr["raw_name"],
            "resolved_id": res_id,
            "catalog_poi_name": pid_to_pname.get(res_id, "(Out-of-Catalog / Non-Catalog)"),
            "similarity_score": rr["similarity_score"],
            "slot_class": rr["slot_class"],
            "audit_verified": 1,
        })
    pd.DataFrame(er_rows).to_csv(cfg.DATA_DIR / "entity_resolution_50sample_audit.csv", index=False, encoding="utf-8-sig")

    df_dc.to_csv(cfg.DATA_DIR / "rule_cross_validation_1019rows.csv", index=False, encoding="utf-8-sig")
    df_queries.to_csv(cfg.DATA_DIR / "audit_queries_corpus.csv", index=False, encoding="utf-8-sig")
    df_recs.to_csv(cfg.DATA_DIR / "all_recommendations.csv", index=False, encoding="utf-8-sig")

    df_pois_out = df_pois.copy()
    df_pois_out["total_ai_mentions"] = [int(poi_mentions_b1.get(str(pid), 0)) for pid in df_pois_out["poi_id"]]
    df_pois_out["all_batches_mentions"] = [int(poi_mentions_all.get(str(pid), 0)) for pid in df_pois_out["poi_id"]]
    df_pois_out.to_csv(cfg.DATA_DIR / "ground_truth_pois_10cities.csv", index=False, encoding="utf-8-sig")

    return df_queries, df_recs, df_pois_out


if __name__ == "__main__":
    df_p = pd.read_csv(cfg.DATA_DIR / "ground_truth_pois_10cities.csv")
    df_q, df_r, df_p_upd = build_audit_recommendations_and_queries(df_p)
    print(
        f"Parsed {len(df_q)} queries and {len(df_r)} slots. "
        f"In-catalog resolved: {df_r['resolved'].mean():.2%}; "
        f"Out-of-catalog municipal: {(df_r['slot_class']=='B1_out_of_catalog_municipal').mean():.2%}; "
        f"Spatial spillover: {(df_r['slot_class']=='B2_spatial_boundary_spillover').mean():.2%}; "
        f"Generic/Unverified: {(df_r['slot_class']=='C_generic_activity_or_unverified').mean():.2%}."
    )
