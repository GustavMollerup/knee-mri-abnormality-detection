"""Rule-based, multilingual label extraction from free-text knee MRI reports.

``extract(report)`` returns, for each of the 12 targets, a soft score in [0, 1]
plus ``__conf`` (confidence), ``__npos`` and ``__nneg`` (supporting clause counts).

This module is copied verbatim from the Kaggle notebook so it can be imported
and unit-tested on its own; it needs only the standard library.
"""
from __future__ import annotations
import re
import unicodedata
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

_PRE = str.maketrans({'ı': 'i', 'İ': 'i', 'I': 'i', 'ß': 'ss', 'đ': 'd', 'Đ': 'd', 'ø': 'o', 'Ø': 'o', 'æ': 'ae', 'Æ': 'ae'})

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join((ch for ch in text if not unicodedata.combining(ch)))
    text = text.replace('\xad', '')
    text = re.sub('[_\\-/\\\\]+', ' ', text)      # unchanged — this is what eats '-' before _SENT_SPLIT sees it
    text = re.sub('[ \\t]+', ' ', text)
    return text

# '-' removed from this set: normalize() already turns every '-' into a space before this regex ever runs,
# so leaving it here was dead code that could never match anything.
_SENT_SPLIT = re.compile('(?<=[.;!?>•])\\s+|\\n+')

# recognizes '>' / '•' / '-' bullets AND "1." / "2)" style numbered items at the start of a line
_BULLET_START = re.compile(r'^([>•\-]|\d{1,2}[.)])\s*')

def unwrap(text: str) -> str:
    if not isinstance(text, str):
        return ''
    out = []
    for line in text.split('\n'):
        s = line.strip()
        starts_new_item = bool(_BULLET_START.match(s))
        if (out and out[-1]
                and not re.search('[.;:!?>*•]$', out[-1])
                and len(out[-1].split()) >= 4
                and s
                and not s[:1].isupper()
                and not starts_new_item):          # <-- the actual fix: don't fold a new bullet/numbered line into the previous one
            out[-1] = out[-1] + ' ' + s
        else:
            out.append(s)
    return '\n'.join(out)



def clauses(text: str):
    norm = normalize(unwrap(text) if FEATURES['unwrap'] else text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]
    merged = []
    for i, c in enumerate(raw):
        if c.endswith(':') and len(c.split()) <= 14 and (i + 1 < len(raw)):
            merged.append(c + ' ' + raw[i + 1])
        merged.append(c)
    out = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend((p.strip() for p in c.split(',') if len(p.split()) > 2))
    return out
FEATURES = {'unwrap': True, 'directional_negation': True, 'oa_inherit': True, 'graded_pathology': True, 'synovitis_backoff': True}

# Ændret til en præventativ kode som undgår ... Merger all tekst til regex, eller fra regex
def _rx(*alts: str) -> re.Pattern:
    return re.compile('|'.join(f'(?:{a})' for a in alts))

PRE_NEG = _rx('\\bno\\b', '\\bnot\\b', '\\bwithout\\b', '\\bnegative for\\b', '\\babsence\\b', '\\bno evidence\\b', '\\bfree of\\b', '\\bnone\\b', '\\bneither\\b', '\\bnor\\b', '\\bsin\\b', '\\bno hay\\b', '\\bausencia\\b', '\\bausentes?\\b', '\\bno se\\b', '\\bpas de\\b', '\\bsans\\b', '\\baucune?\\b', '\\bgeen\\b', '\\bzonder\\b', '\\bniet\\b', '\\bkeine?[nmrs]?\\b', '\\bohne\\b', '\\bnicht\\b', '\\bkein\\b', '\\bnema\\b', '\\bbez\\b', '\\bnisu\\b', '\\bnije\\b', '\\bδεν\\b', '\\bχωρις\\b', 'ουδεν', '\\bουτε\\b', '\\bбез\\b', '\\bне\\b', 'липсва', '\\bняма\\b')
POST_NEG = _rx('\\byok\\b', '\\byoktur\\b', 'izlenmemekte', 'saptanmadi', '\\bdegil\\b', 'gozlenmemekte', 'mevcut degil', 'eslik etmiyor', '\\bizlenmedi\\b', 'izlenmemistir', 'saptanmamistir', 'gorulmemistir', '\\bnema znakova\\b', 'bez znakova')
NEGATION = _rx(PRE_NEG.pattern, POST_NEG.pattern, '\\bunremarkable\\b')
NEG_WINDOW = 90

# Tjekker om det modsatte er gældende; altså, hvis der står ingen skade (tilføjet franske 'mais')
def _negated(clause: str, start: int, end: int) -> bool:
    for m in PRE_NEG.finditer(clause):
        if m.end() <= start and start - m.end() <= NEG_WINDOW:
            if not re.search('\\b(but|however|ancak|mais|fakat|pero|maar|aber|no i|ali|ωστοσο|αλλα|но)\\b', clause[m.end():start]):
                return True
    for m in POST_NEG.finditer(clause):
        if m.start() >= end and m.start() - end <= NEG_WINDOW:
            return True
    return False
NORMALITY = _rx('\\bnormal', '\\bintact\\b', '\\bpreserved\\b', '\\bwithin normal limits\\b', 'limites normales', '\\bconservad', '\\bintegr', '\\bnormales\\b', '\\bdoga(l|ll)\\b', 'korunmus', '\\bnormaldir\\b', 'olagan', '\\buredn', '\\bocuvan', '\\bodrzan', '\\bintakt', '\\bprimjeren', '\\bodrzanog kontinuiteta', '\\bodržan', 'φυσιολογικ', 'ακεραι', 'δεν παρατηρουνται', 'δεν σημειωνονται', 'unauffallig', 'regelrecht', '\\bo\\.?b\\.?\\b', 'нормал', 'запазен', 'съхранен', '\\bбез особености\\b', 'интактн', '\\bgaaf\\b', '\\bnormaal\\b')
NORMAL_PHRASE = _rx('\\bsin alteracion', '\\bsin cambios\\b', '\\bsin particularidad', '\\bsin hallazgos\\b', '\\bsin lesion', '\\bsin signos de (rotura|lesion)', '\\bcontinu[oa]s?\\b', '\\bcontinuidad conservada\\b', '\\bno abnormalit', '\\bno significant abnormalit', '\\bunremarkable\\b', '\\bno evidence of (tear|injury|abnormalit)', '\\bohne auffalligkeit', '\\bkein nachweis\\b', '\\bohne befund\\b', '\\bgeen afwijking', '\\bzonder afwijking', '\\bsans anomalie', "\\bpas d[e']anomalie", '\\bbez osobitosti\\b', '\\bbez znakova (rupture|lezije)\\b', '\\bbez patoloskih\\b', 'χωρις αλλοιωσ', 'χωρις παθολογ', 'δεν παρατηρουνται (αξιολογα|παθολογ)', '\\bбез особености\\b', '\\bбез патологич', '\\bбез данни за\\b', '\\bozel bir ozellik yok', '\\bpatolojik bulgu (yok|izlenmemis)')
UNCERTAIN = _rx('\\bpossible\\b', '\\bprobable\\b', '\\bsuspicious\\b', '\\bsuspected?\\b', 'cannot (be )?exclude', '\\bmay\\b', '\\bquestionable\\b', '\\bequivocal\\b', '\\br/o\\b', '\\bdd\\b', '\\blikely\\b', '\\bsuggest', '\\bcompatible with\\b', '\\bposible\\b', 'sin criterios categoricos', '\\bdudos', '\\bsugier', '\\bmuhtemel\\b', '\\bolasi\\b', '\\bsupheli\\b', '\\bizlenim', '\\bdusundur', '\\bmoguce\\b', '\\bvjerojatno\\b', '\\bsumnja\\b', '\\bmoze odgovarati\\b', 'πιθαν', 'υποπτ', '\\bmoglich', '\\bverdachtig', '\\bfraglich', '\\bv\\.?a\\.?\\b', '\\bwohl\\b', '\\bвъзможно\\b', '\\bвероятно\\b', 'суспект', '\\bmogelijk\\b', '\\bverdacht\\b')


TEAR = _rx('\\btear', '\\btorn\\b', '\\brupture', '\\bdisruption\\b', 'discontinuit', '\\bavuls', '\\bmacerat', '\\bbuckethandle\\b', 'bucket handle', '\\brotura\\b', '\\broturas\\b', '\\bruptura', '\\bdesgarro', '\\broto\\b', '\\bdechirure', '\\bdechire', '\\bscheur', '\\bruptuur', 'gescheurd', '\\briss\\b', 'einriss', '\\bruptur', 'zerreiss', '\\blasion', '\\bausriss', '\\byirtik', '\\byirtig', '\\bkopma\\b', 'butunluk kaybi', '\\brupturu\\b', 'devamsizlik', '\\brupture\\b', '\\bdevamliligi secilememis', '\\bpuknuce', '\\bprekid\\b', '\\bpukotin', '\\bruptur', 'ρηξη', 'ρηξις', 'ρηγμα', 'ασυνεχεια', 'руптура', 'разкъсв', 'разрив', 'скъсв', '\\bлезия\\b')
DEGEN = _rx('degenerat', '\\bmucoid\\b', '\\bmyxoid\\b', '\\bfray', '\\bfissur', 'dejeneratif', '\\bmukoid\\b', 'degenerativn', 'εκφυλ', 'дегенерат', '\\bμυξοειδ', '\\bμυξωδ', '\\bmeniskopat', '\\bmeniscopath', '\\bmuco ?ide\\b', 'aufgefasert', '\\bdejenerasyon\\b')
INJURY = _rx('\\binjur', '\\bsprain', '\\blesion', '\\blasion', '\\bedema\\b', '\\boedema\\b', '\\bodem\\b', '\\bedem\\b', '\\bοιδημα', '\\bодем', '\\bедем', '\\bstrain\\b', '\\bhigh signal\\b', '\\bsignal alteration\\b', '\\bhiperintens', '\\bhyperintens', 'aumento de senal', 'alteracion de senal', 'cambio de senal', '\\bsignalanhebung', '\\bsignalalteration', 'verhoogd signaal', 'sinyal artis', 'αυξημενο σημα', 'повишен сигнал', '\\besguince\\b', '\\bthicken', '\\bzadebljanje\\b', '\\bverdikking\\b', '\\bdistenzij', '\\blaksite\\b', '\\blaxity\\b', '\\bpartial\\b', '\\bparcijaln', '\\bparcial', '\\bpartiel', '\\bpartiell')
_GRADE_RX = re.compile('(?:grade|grad|grado|grau|derece|stupnja|stupanj|βαθμ|степен|icrs|outerbridge)[\\s:]*(?:grade\\s*)?([1-4]|iv|iii|ii|i)\\b')
_ROMAN = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4}

def _grade_of(clause: str):
    best = None
    for m in _GRADE_RX.finditer(clause):
        v = m.group(1)
        n = _ROMAN.get(v, None) if not v.isdigit() else int(v)
        if n is not None and (best is None or n > best):
            best = n
    return best
ANAT = {'ACL': _rx('anterior cruciate', '\\bacl\\b', 'cruzado anterior', '\\blca\\b', 'croise anterieur', 'voorste kruisband', '\\bvkb\\b', 'vorderes kreuzband', 'vorderen kreuzband', 'vordere kreuzband', 'on capraz', '\\bocb\\b', 'anterior capraz', 'prednji krizni', 'prednjeg krizn', 'προσθι[οα][^ ]* χιαστ', 'προσθιου χιαστου', 'χιαστο[^ ]* συνδεσμ', '\\bχιαστ\\w*', 'предна кръстна', 'предната кръстна', 'предна кръста', 'cruciate ligaments', 'ligamentos cruzados', 'ligaments croises', 'kruisbanden', 'kreuzbander', 'capraz baglar', 'krizn[a-z]* ligament[a-z]*', 'χιαστοι συνδεσμ', 'χιαστων συνδεσμ', 'кръстните връзки', 'кръстни връзки'), 'MCL': _rx('medial collateral', '\\bmcl\\b', 'tibial collateral', 'colateral medial', 'colateral interno', '\\blcm\\b', 'collateral medial', 'collateral interne', 'mediale collaterale', 'binnenband', '\\b(mediale|laterale) banden\\b', '\\bcollaterale banden\\b', 'innenband', 'mediales? kollateral', '\\bic yan bag', 'medial kollateral', '\\biyb\\b', 'medyal kollateral', 'medijalni kolateraln', 'medijalnog kolateraln', 'εσω πλαγι', 'εσωτερικο πλαγι', '\\bπλαγι\\w* συνδεσμ', '\\bπλαγιοι\\b', 'медиален колатерал', 'вътрешна странична', '\\bколатерал\\w*', '\\bcolaterales\\b', '\\bcollateraux\\b', '\\bcollateralen\\b', '\\bkolateralni\\b', 'collateral ligaments', 'ligamentos colaterales', 'ligaments collateraux', 'collaterale banden', 'kollateralbander', 'seitenbander', 'yan baglar', 'kolateraln[a-z]* ligament[a-z]*', 'πλαγιοι συνδεσμ', 'πλαγιων συνδεσμ', 'колатерални връзки', 'страничните връзки'), 'Medial Meniscus': _rx('medial meniscus', '\\bmm\\b(?= tear)', 'medial menisc', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'mediale meniscus', 'binnenmeniscus', 'innenmeniskus', 'medialen? meniskus', 'innenmeniskushinterhorn', 'medyal menisk', '\\bic menisk', 'medijalni meniskus', 'medijalnog meniskusa', 'medijalnom meniskusu', 'medijaln\\w* menisk\\w*', '\\bmedijalnog meniska\\b', 'medijalni menisk', 'εσω μηνισκ', 'μηνισκ[^ ]* του εσω', 'εσω διαμερισμα[^.]{0,40}μηνισκ', 'медиалния менискус', 'медиален менискус', 'вътрешния менискус', 'oba meniska', 'both menisci', 'ambos meniscos', 'beide menisci', 'her iki menisku', 'amfoteroi\\w* mhnisk', 'αμφοτερ\\w* μηνισκ', 'двата менискуса', 'medial (and|&) lateral menisc'), 'Lateral Meniscus': _rx('lateral meniscus', 'lateral menisc', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'laterale meniscus', 'buitenmeniscus', 'aussenmeniskus', 'lateralen? meniskus', 'aussenmeniskushinterhorn', 'lateral menisk', '\\bdis menisk', 'lateralni meniskus', 'lateralnog meniskusa', 'lateralnom meniskusu', 'lateraln\\w* menisk\\w*', '\\blateralnog meniska\\b', 'εξω μηνισκ', 'μηνισκ[^ ]* του εξω', 'εξω διαμερισμα[^.]{0,40}μηνισκ', 'латералния менискус', 'латерален менискус', 'външния менискус', 'oba meniska', 'both menisci', 'ambos meniscos', 'beide menisci', 'her iki menisku', 'αμφοτερ\\w* μηνισκ', 'двата менискуса', 'medial (and|&) lateral menisc')}
# OA_EVIDENCE = _rx('osteoarthrit', '\\barthros', '\\bgonarthros', '\\bosteoarthros', 'chondropath', 'chondromalac', 'condropat', 'condromalac', '\\bchondros', '\\bchondrosis\\b', 'chondral (loss|defect|ulcer|thinning|injury|fissur|wear)', 'cartilage (loss|thinning|defect|fissur|wear|damage|heterogeneity|irregularit)', '(loss|thinning|fissur|defect|ulcer|erosion|denudation) of[^.]{0,20}cartilage', 'articular cartilage[^.]{0,30}(loss|thin|fissur|defect|erosion|wear|irregular)', 'osteophyt', 'osteofit', 'osteofyt', 'osteofito', 'osteophyten', 'spurring', 'joint space narrowing', 'pinzamiento articular', 'reduced joint space', 'kikirdak kayb', 'kikirdak incelme', 'kondropati', 'kondral', 'kikirdak dejener', 'eklem aralig\\w* daral', 'eklem mesafesi daral', 'kikirdak kalinlig\\w* azal', 'kraakbeen', 'gonartrose', 'artrose', '\\bknorpel', 'arthrose', 'gonarthrose', 'hrskavic', 'hondromalac', 'artroz', 'osteoartrit', 'artrotsk', 'artrotick', '\\boa promjen', '\\boa\\b', 'degenerativne promjene hrskav', 'χονδρ[^ ]*παθ', 'αρθριτ', 'αρθρωσ', 'οστεοφυτ', 'χονδρομαλακ', 'αρθρικου χονδρου', 'εξαλειψη του αρθρικου χονδρου', 'διαβρωση του αρθρικου χονδρ', 'λεπτυνση[^.]{0,30}χονδρ', 'φθορα[^.]{0,20}χονδρ', 'артроз', 'хондропат', 'остеофит', 'хрущял[^.]{0,40}(изтън|увред|дефект|липс)', 'изтъняване[^.]{0,30}хрущял', 'хондромалац', 'ulcera[s]? condral', 'cartilago[^.]{0,25}(perdida|adelgaz)', 'icrs grade', 'icrs\\b', 'outerbridge', '\\bdenudation\\b', 'denudacij', 'erozivne promjene', '\\berosion of[^.]{0,20}cartilage', 'kraakbeenlijden', 'kraakbeenverlies')
OA_STRONG = _rx('osteoarthrit', '\\barthros', '\\bgonarthros', '\\bosteoarthros', 'gonartrose', 'artrose', 'arthrose', 'gonarthrose', 'artroz', 'osteoartrit', 'artrotsk', 'artrotick', '\\boa promjen', '\\boa\\b', 'артроз', 'osteophyt', 'osteofit', 'osteofyt', 'osteofito', 'osteophyten', 'οστεοφυτ', 'остеофит')
OA_MODERATE = _rx('joint space narrowing', 'pinzamiento articular', 'reduced joint space', 'eklem aralig\\w* daral', 'eklem mesafesi daral', 'chondral (loss|thinning|wear)', 'cartilage (loss|thinning|wear)', '(loss|thinning|erosion|denudation) of[^.]{0,20}cartilage', 'articular cartilage[^.]{0,30}(loss|thin|erosion|wear)', '\\berosion of[^.]{0,20}cartilage', '\\bdenudation\\b', 'kikirdak kayb', 'kikirdak incelme', 'kikirdak dejener', 'kikirdak kalinlig\\w* azal', 'degenerativne promjene hrskav', 'denudacij', 'cartilago[^.]{0,25}(perdida|adelgaz)', 'kraakbeenverlies', 'εξαλειψη του αρθρικου χονδρου', 'διαβρωση του αρθρικου χονδρ', 'λεπτυνση[^.]{0,30}χονδρ', 'φθορα[^.]{0,20}χονδρ', 'хрущял[^.]{0,40}(изтън|липс)', 'изтъняване[^.]{0,30}хрущял')
OA_WEAK = _rx('chondropath', 'chondromalac', 'condropat', 'condromalac', '\\bchondros', '\\bchondrosis\\b', 'kondropati', 'hondromalac', 'χονδρ[^ ]*παθ', 'χονδρομαλακ', 'хондропат', 'хондромалац', 'chondral (defect|ulcer|injury|fissur)', 'cartilage (defect|fissur|damage|heterogeneity|irregularit)', '(fissur|defect|ulcer) of[^.]{0,20}cartilage', 'articular cartilage[^.]{0,30}(fissur|defect|irregular)', 'ulcera[s]? condral', 'хрущял[^.]{0,40}(увред|дефект)', 'kraakbeenlijden', 'kraakbeen', '\\bknorpel', 'hrskavic', 'kondral', 'αρθρικου χονδρου', 'icrs grade', 'icrs\\b', 'outerbridge', 'spurring', 'αρθριτ', 'αρθρωσ', 'erozivne promjene')
TF_SITE = _rx('compartment', 'compartimento', 'compartiment', 'kompartman', 'kompartiment', 'kompartment', 'odjelj', 'διαμερισμα', 'компартм', '\\bотдел', 'femorotibial', 'tibiofemoral', 'femoro tibial', 'femorotibiaal', 'femorotibijaln', 'феморотибиал', '\\bft zglob', 'tibiofemoraln', 'condyle', 'condilo', 'kondyl', 'kondil', 'condyl', 'κονδυλ', 'кондил', '\\bplateau', '\\bplato\\b', 'platillo', 'meseta', 'плато', 'tibiaplateau', 'tibijaln\\w* plato', 'tibyal plato', 'tibia plato', 'κνημιαι', 'μηριαι', 'weightbearing', 'weightbaring', 'zona de carga', 'dragende deel', 'agirlik tasiyan', '\\bfemur\\b', '\\btibia\\b', '\\bfemoral\\b', '\\btibial\\b', '\\bfemura\\b', '\\btibije\\b', '\\bmesarthrio\\b', 'μεσαρθριο')
PF_SITE = _rx('patellofemoral', 'femoropatellar', 'femoropatelar', 'patelofemoral', 'retropatellar', 'retrorotulian', 'trochlea', 'troclea', 'troklea', 'trochlear', 'trohlej', 'τροχιλ', '\\bpatella', '\\bpatellar', 'rotulian', '\\brotula\\b', '\\bpatele\\b', 'patellofemoraal', 'femoropatellair', 'επιγονατιδ', 'μηροεπιγονατιδ', 'пател', 'феморопател', 'anterior compartment', 'compartimento anterior', 'prednj\\w* odjeljk', '\\bfp zglob', '\\bpf zglob', '\\bfaset', '\\bfacet', 'patellofemoraln')
SIDE_MEDIAL = _rx('\\bmedial\\w*', '\\bmedyal\\w*', '\\bmedijaln\\w*', '\\bmediaal\\w*', '\\bmediale\\w*', '\\binterno\\b', '\\binterna\\b', '\\binternos\\b', '\\binterne\\b', '\\binnen\\w*', '\\bic\\b', '\\bunutarnj\\w*', '\\bεσω\\w*', '\\bεσωτερικ\\w*', '\\bмедиал\\w*', '\\bвътреш\\w*', '\\bbinnen\\w*', '\\bmediaal\\b', '\\bmediales?\\b')
SIDE_LATERAL = _rx('\\blateral\\w*', '\\bexterno\\b', '\\bexterna\\b', '\\bexternos\\b', '\\bexterne\\b', '\\bdis\\b', '\\blateraln\\w*', '\\baussen\\w*', '\\bbuiten\\w*', '\\bεξω\\w*', '\\bεξωτερικ\\w*', '\\bлатерал\\w*', '\\bвъншн\\w*', '\\bvanjsk\\w*')
SIDE_ANTERIOR = _rx('\\banterior\\w*', '\\bant\\b', '\\bon\\b', '\\bprednj\\w*', '\\bvorder\\w*', '\\bvoorste\\b', '\\bπροσθι\\w*', '\\bпредн\\w*', '\\banteriyor\\w*', '\\bavant\\b', '\\banterieur\\w*')
GLOBAL_OA = _rx('tri ?compartment', 'all three compartment', 'global(ised)? (oa|osteoarthrit)', '\\bgonarthros', '\\bgonartros', '\\bgonarthrose', '\\bgonartrose', 'gonartro', 'goanrtrot', 'gonartrot', 'osteoarthritis of the knee', 'artrosis (de |)(la )?rodilla', 'knee osteoarthrit', '\\bdiz osteoartrit', '\\bgonartroz', 'artroza koljena', 'οστεοαρθριτιδα', 'αρθριτιδα του γονατος', 'εκφυλιστικη οστεοαρθριτ', 'артроза на колянната', 'гонартроз', 'degenerative joint disease', '\\bdjd\\b', 'three compartments', 'compartmens', 'compartments')
DIRECT = {'Effusion': _rx('\\beffusion', 'joint fluid', 'intra ?articular fluid', '\\bhydrops\\b', '\\bhemarthros', '\\bhaemarthros', 'derrame articular', '\\bderrame\\b', 'liquido articular', 'hemartrosis', 'epanchement', 'gewrichtsvocht', '\\bvocht\\b', 'gewrichtseffusie', 'opzetting van suprapatell', 'gelenkerguss', '\\berguss\\b', 'gelenksergu', 'gelenksflussigkeit', 'eklem\\w* ic\\w* sivi', 'efuzyon', 'eklem sivisi', 'eklem mesafesinde sivi', 'sivi (miktari|artisi|birikimi)', 'sivi artis', '\\bsivi\\b[^.]{0,25}artmis', '\\bizljev', '\\bizliv', 'zglobn[^ ]* tekucin', '\\bhidrops\\b', 'αρθρικ[^ ]* υγρ', 'υγρου ενδαρθρικα', 'ενδαρθρικ[^ ]* υγρ', 'ποσοτητα υγρου', 'ενδαρθρικ', 'αρθρικη συλλογη', 'υγρο στην αρθρωση', 'υγρου στην αρθρωση', 'συλλογη υγρου', 'ενθαρθρικ', 'ставен излив', 'излив', 'ставна течност', 'синовиална течност'), 'Synovitis': _rx('synovit', 'sinovit', 'synovial (thickening|proliferation|hypertroph)', 'thicken\\w* synovial', 'hypertroph\\w* of the synovium', 'synoviale? (verdikking|proliferatie)', 'verdikkingen van (het )?synovium', 'synovialitis', 'synovialis(verdickung|proliferation)', 'reizsynovial', 'sinovijalitis', 'sinovitis', 'zadebljanje sinovij', 'proliferacij\\w* sinovij', 'sinovijaln\\w* proliferacij', 'υμενιτιδα', 'συνοβιτιδα', 'υμενικ[^ ]* υπερτροφ', 'αρθρικου υμεν', 'παχυνση[^.]{0,20}υμεν', 'υμενα', 'синовит', 'синовиал[^ ]* (задебел|пролифер)', '\\bpannus\\b', '\\bhoffit', 'sinovyal\\w* (kalinlas|proliferas)', 'sinovyal hipertrof', '\\bartrit\\b', '\\barthritis\\b'), "Baker's": _rx('baker', 'popliteal cyst', 'quiste popliteo', 'quistes popliteos', 'kyste poplite', 'popliteale? cyst', 'poplitealzyste', 'bakerzyste', 'popliteal kist', '\\bbakerova\\b', 'poplitealn[^ ]* cist', 'popliteal\\w* cist', 'κυστη baker', 'πολυχωρη συνοβιακη κυστη', 'κυστη του baker', 'συνοβιακη κυστη', 'κυστη τυπου baker', 'киста на бейкър', 'бейкърова киста', 'поплитеална киста', 'бекеров', 'gastrocnemio ?semimembranos', 'gastrocnemius semimembranosus burs'), 'Contusion': _rx('\\bcontusion', 'bone bruise', 'bone marrow (o?edema|contusion)', 'marrow o?edema', '\\bkontuz', 'medular bone o?edema', 'osseous contusion', 'contusion osea', 'edema oseo', 'edema de medula osea', 'contusiones oseas', 'oedeme osseux', 'contusion osseuse', 'botcontusie', 'botoedeem', 'beenmergoedeem', 'botmergoedeem', 'knochenmarkodem', 'knochenodem', 'knochenmarksodem', 'kontusion', 'kemik kontuzyonu', 'kemik iligi odemi', 'kemik odemi', 'kemik iliginde odem', 'kontuzyonel kemik', 'kemik iligi odemleri', 'kostani edem', 'edem kosti', 'kontuzij', 'kostane srzi[^.]{0,20}edem', 'οστεομυελικ[^ ]* οιδημα', 'οστικο οιδημα', 'μυελικο οιδημα', 'οστικο μωλωπ', 'костномозъчен едем', 'костен едем', 'контузионен', 'костно мозъчен едем'), 'Fracture': _rx('\\bfractur', '\\bfract\\b', '\\bfractura', '\\bfracturas\\b', '\\bfractuur', '\\bbreuk\\b', '\\bfraktur', '\\bbruch\\b', '\\bkirik\\b', '\\bkirigi\\b', '\\bkiri[kg]\\w*', '\\bprijelom', 'impresijsk[^ ]* fraktur', 'impaktcij', 'καταγμα', 'καταγματ', 'фрактур', 'счупван', 'фисур', 'insufficiency fracture', 'stress fracture', 'avulsion fracture', 'subchondral fracture', 'subkondral kiri', 'impaction (fracture|injury)', 'osteochondral (fracture|impaction)', '\\bsegond\\b', 'impactiefractuur', 'subchondrale impression', 'subchondraler? impress')}
DECOY = {'Fracture': _rx('microfractur', '\\bfracture (risk|prophyla)'), "Baker's": _rx('meniscal cyst', 'quiste meniscal', 'parameniscal')}
PAIRED = {'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus'}
OA_TARGETS = ['Medial OA', 'Lateral OA', 'PF OA']
PLURAL_MENISCI = _rx('\\bmenisci\\b', '\\bmeniscos\\b', '\\bmenisques\\b', '\\bmenisken\\b', '\\bmeniskusi\\b', '\\bmenisk\\w*ler\\b', '\\bμηνισκοι\\b', '\\bμηνισκων\\b', '\\bменискуси\\b', '\\bменискусите\\b', '\\bmenisci\\w*\\b')
ANY_SIDE = _rx(SIDE_MEDIAL.pattern, SIDE_LATERAL.pattern)
STEM_MENISCUS = _rx('menisc\\w*', 'menisk\\w*', 'μηνισκ\\w*', 'мениск\\w*')
STEM_CRUCIATE = _rx('cruciate', 'cruzado', 'croise', 'kruisband', 'kreuzband', 'capraz bag\\w*', 'krizn\\w*', 'χιαστ\\w*', 'кръстн\\w*', '\\bacl\\b', '\\blca\\b', '\\bvkb\\b', '\\bocb\\b', '\\bacb\\b')
STEM_COLLATERAL = _rx('collateral\\w*', 'colateral\\w*', 'kollateral\\w*', 'collaterale\\w*', 'kolateraln\\w*', 'yan bag\\w*', 'πλαγι\\w*', 'колатерал\\w*', 'странич\\w*', 'innenband\\w*', 'binnenband\\w*', '\\bmcl\\b', '\\blcm\\b', '\\biyb\\b')
STEM_FRACTURE = _rx('fractur\\w*', 'fraktur\\w*', 'fractuur\\w*', '\\bfract\\b', 'kiri[kgğ]\\w*', 'prijelom\\w*', 'lom kosti', '\\bbreuk\\w*', '\\bbruch\\w*', 'καταγμα\\w*', 'καταγματ\\w*', 'фрактур\\w*', 'счупван\\w*', 'fisur\\w* (osea|oseas|kost)', 'fissur\\w* kost')


def _near(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int=55):
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        if qual_rx.search(clause[lo:hi]):
            return True
    return False
STEM_RULES = {'ACL': (STEM_CRUCIATE, SIDE_ANTERIOR), 'MCL': (STEM_COLLATERAL, SIDE_MEDIAL), 'Medial Meniscus': (STEM_MENISCUS, SIDE_MEDIAL), 'Lateral Meniscus': (STEM_MENISCUS, SIDE_LATERAL)}

class _Matcher:

    def __init__(self, phrase_rx, stem=None, side=None, window=55):
        self.phrase_rx = phrase_rx
        self.stem = stem
        self.side = side
        self.window = window

    def search(self, clause):
        m = self.phrase_rx.search(clause)
        if m is not None:
            return m
        if self.stem is not None and _near(clause, self.stem, self.side, self.window):
            return self.stem.search(clause)
        return None
ANAT_MATCH = {t: _Matcher(ANAT[t], *STEM_RULES[t]) for t in PAIRED}
DIRECT_MATCH = {t: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if t == 'Fracture' else rx) for t, rx in DIRECT.items()}
SEV_LOW = _rx('\\bsmall\\b', '\\bminimal\\b', '\\btrace\\b', '\\bmild\\b', '\\bslight\\b', '\\btiny\\b', '\\bscant\\b', '\\bdiscrete\\b', '\\blow ?grade\\b', '\\bincipient\\b', '\\bleve\\b', '\\bminim', '\\bpeque', '\\bfina\\b', '\\bfino\\b', '\\bligero\\b', '\\bescaso\\b', '\\bdiscreto\\b', '\\bhafif\\b', '\\baz miktarda\\b', '\\bsilik\\b', '\\bmanj\\w*', '\\bblago\\b', '\\bdiskretn', '\\bmalo\\b', '\\bpocetn', '\\bgering', '\\bdiskret', '\\bkleine?r?\\b', '\\bwenig\\b', '\\bzarte?\\b', '\\bbeperkte?\\b', '\\bgeringe\\b', '\\bweinig\\b', '\\blichte?\\b', '\\blicht\\b', '\\bηπι', '\\bμικρ', '\\bελαχιστ', '\\bαρχομεν', '\\bминимал', '\\bлек', '\\bмалк', '\\bнеголям')
SEV_HIGH = _rx('\\blarge\\b', '\\bmarked\\b', '\\bmassive\\b', '\\bsevere\\b', '\\bextensive\\b', '\\bmoderate\\b', '\\bgross\\b', '\\bsignificant\\b', '\\babundant\\b', '\\btense\\b', '\\bcomplete\\b', '\\bfull ?thickness\\b', '\\bhigh ?grade\\b', '\\badvanced\\b', '\\bmoderad', '\\bimportante\\b', '\\bsevera?\\b', '\\bmarcad', '\\bcuantios', '\\bespesor total\\b', '\\bcompleta?\\b', '\\bbelirgin\\b', '\\byaygin\\b', '\\bileri\\b', '\\bciddi\\b', '\\bbol\\b', '\\bkomplet', '\\bopsezan\\b', '\\bveliki\\b', '\\bizrazit', '\\bznacajn', '\\bumjeren', '\\buznapredoval', '\\bpotpun', '\\bkompleksn', '\\bausgepragt', '\\bdeutlich', '\\bmassiv', '\\bmassig', '\\bgross', '\\buitgebreid', '\\bgevorderd', '\\bveel\\b', '\\bmatige?\\b', '\\bvolledig', '\\bμετρι', '\\bμεγαλ', '\\bεκτεταμεν', '\\bευμεγεθ', '\\bσοβαρ', '\\bπληρη', '\\bголям', '\\bизразен', '\\bзначим', '\\bумерен', '\\bобилен', '\\bпълн')
GRADE_HIGH = re.compile('grade?[ao]?\\s*(3|4|iii|iv)\\b|icrs grade (iii|iv|3|4)|stupnja iv|stupnja iii|\\bgrado (3|4)\\b|\\bgrad (3|4)\\b|\\bgrade (3|4)\\b')
DEGENERATIVE_MARROW = _rx('subchondral', 'subcondral', 'subkondral', 'supkondraln', 'subchondraln', 'υποχονδρι', 'υπαρθρικ', 'субхондрал', 'subchondrale?', 'subartikuler', '\\bcyst', '\\bquist', '\\bzyste\\b', '\\bcistic', 'reactive', 'reactivo', 'degenerative', 'degenerativ', 'reaktiv', '\\bcisti\\b')
TRAUMA = _rx('\\bbruise\\b', '\\bcontusion', '\\bkontuz', '\\btrauma', '\\bimpaction\\b', '\\bpivot shift\\b', '\\bkissing\\b', '\\bacute\\b', '\\bagudo\\b', '\\bakut', '\\bpivot kaymasi\\b', '\\bcontusion osseuse\\b', '\\bbone bruise\\b', '\\bbotcontusie\\b', '\\bконтузион', '\\bμωλωπ', '\\bkontuzij', '\\bimpaktcij', '\\bimpakcij', '\\bfall\\b', '\\binjury\\b', '\\bimpression\\b')
SYNOVIAL_PROXY = _rx('bursit', 'burzit', '\\bbursa\\b[^.]{0,30}(fluid|distend|sivi|tekucin|opzetting)', 'suprapatellar (bursitis|effusion|recess)', 'suprapatellar bursa', 'suprapatellar bursada', 'suprapatelarno', 'suprapatellaire recessus', 'hoffa', 'hoffit', 'plica', 'plika', 'πλικα', 'fat pad[^.]{0,20}(edema|oedema)', 'kapsul', 'capsul', 'καψ', 'капсул', '\\bpannus\\b', '\\bsinov', '\\bsynov')

def _polarity(clause: str, span=None) -> str:
    if UNCERTAIN.search(clause):
        return 'uncertain'
    if span is None or not FEATURES['directional_negation']:
        if NEGATION.search(clause):
            return 'negative'
    elif _negated(clause, span[0], span[1]):
        return 'negative'
    if NORMALITY.search(clause):
        if TEAR.search(clause) or GRADE_HIGH.search(clause):
            return 'positive'
        return 'negative'
    return 'positive'

def _severity(clause: str) -> float:
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and (not low):
        return 1.0
    if low and (not high):
        return 0.45
    if high and low:
        return 0.8
    return 0.75

def _grade(n_pos, n_neg, n_unc, best):
    if n_pos or n_unc:
        score = min(0.97, 0.5 + 0.45 * best + 0.015 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.2 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = (0.28, 0.05)
    return (score, conf)

def _paired_weight(clause: str, meniscus: bool) -> float:
    g = _grade_of(clause) if FEATURES['graded_pathology'] else None
    tear = TEAR.search(clause) is not None
    if meniscus:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.95 if g >= 3 else 0.3
        elif DEGEN.search(clause):
            base = 0.35
        else:
            base = 0.45
    elif tear:
        base = 1.0
    elif g is not None:
        base = 0.85 if g >= 2 else 0.3
    elif DEGEN.search(clause):
        base = 0.4
    else:
        base = 0.55
    if SEV_HIGH.search(clause) and (not SEV_LOW.search(clause)):
        base = min(1.0, base * 1.2)
    elif SEV_LOW.search(clause) and (not SEV_HIGH.search(clause)):
        base *= 0.7
    return base

def _score_paired(cls, tgt):
    anat_rx = ANAT_MATCH[tgt]
    path_rx = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)
    meniscus = 'Meniscus' in tgt
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        hit = anat_rx.search(c)
        if hit is None and meniscus and PLURAL_MENISCI.search(c) and (not ANY_SIDE.search(c)):
            hit = PLURAL_MENISCI.search(c)
        if hit is None:
            continue
        pm = path_rx.search(c)
        if pm is None and _grade_of(c) is None:
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c) and (not NEGATION.search(c))):
                n_neg += 1
            continue
        span = (pm.start(), pm.end()) if pm is not None else None
        pol = _polarity(c, span)
        if pol == 'positive':
            n_pos += 1
            best = max(best, _paired_weight(c, meniscus))
        elif pol == 'negative':
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.45 * _paired_weight(c, meniscus))
    s, cf = _grade(n_pos, n_neg, n_unc, best)
    return (s, cf, n_pos, n_neg)

def _score_clauses(cls, anat_rx, path_rx=None, decoy_rx=None, context_penalty=None, context_bonus=None):
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if path_rx is not None and (not path_rx.search(c)):
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c) and (not NEGATION.search(c))):
                n_neg += 1
            continue
        pol = _polarity(c, (m.start(), m.end()))
        if pol == 'positive':
            n_pos += 1
            w = _severity(c)
            if context_penalty is not None and context_penalty.search(c):
                w *= 0.45
            if context_bonus is not None and context_bonus.search(c):
                w = min(1.0, w * 1.35)
            best = max(best, w)
        elif pol == 'negative':
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.3)
    s, c = _grade(n_pos, n_neg, n_unc, best)
    return (s, c, n_pos, n_neg)

_STRICT_OA_TARGETS = {'Medial OA', 'Lateral OA'}
# High-severity wording that corroborates WEAK OA evidence for the strict targets.
_SEV_TEXT_HIGH = SEV_HIGH
# PF OA is deliberately excluded from the strict/corroboration requirement: validated
# against gold, PF OA's own labels count bare, uncorroborated patellar chondral findings
# ("condropatia rotuliana", explicitly "low grade"/"mild" patellar cartilage language) as
# positive. Gating it the same as Medial/Lateral OA cost 7-10 of its 21 true positives
# with no matching over-firing problem to justify it.

def _score_oa(cls):
    acc = {t: {'pos': 0, 'neg': 0, 'unc': 0, 'best': 0.0} for t in OA_TARGETS}
    g_pos, g_neg, g_best = (0, 0, 0.0)
    # Marks a compartment as "seen" even when the evidence found there was too weak to
    # count -- stops the tricompartmental g_pos/oa_inherit fallback below from filling
    # that compartment in from an unrelated global finding elsewhere in the report.
    touched = {t: False for t in OA_TARGETS}

    for c in cls:
        strong_m = OA_STRONG.search(c)
        mod_m = None if strong_m else OA_MODERATE.search(c)
        weak_m = None if (strong_m or mod_m) else OA_WEAK.search(c)
        m = strong_m or mod_m or weak_m
        if m is None:
            continue

        corroborated = None
        if weak_m is not None:
            g = _grade_of(c)
            corroborated = (g is not None and g >= 2) or bool(GRADE_HIGH.search(c)) or bool(_SEV_TEXT_HIGH.search(c))

        pol = _polarity(c, (m.start(), m.end()))
        sev = _severity(c)
        tf_med = _near(c, TF_SITE, SIDE_MEDIAL, 45)
        tf_lat = _near(c, TF_SITE, SIDE_LATERAL, 45)
        pf = PF_SITE.search(c) is not None

        def _counts(target):
            if weak_m is None:
                return True  # STRONG or MODERATE: always counts
            if target in _STRICT_OA_TARGETS:
                return bool(corroborated)
            return True  # PF OA: bare WEAK evidence counts too

        hits = []
        if tf_med: hits.append('Medial OA')
        if tf_lat: hits.append('Lateral OA')
        if pf: hits.append('PF OA')

        if not hits:
            if weak_m is not None:
                continue  # unsited + uncorroborated-tier evidence: too thin to inherit anywhere
            if pol == 'positive':
                g_pos += 1
                g_best = max(g_best, sev if GLOBAL_OA.search(c) else sev * 0.7)
            elif pol == 'negative':
                g_neg += 1
            continue

        for t in hits:
            touched[t] = True
            if not _counts(t):
                continue
            if pol == 'positive':
                acc[t]['pos'] += 1
                acc[t]['best'] = max(acc[t]['best'], sev)
            elif pol == 'negative':
                acc[t]['neg'] += 1
            else:
                acc[t]['unc'] += 1
                acc[t]['best'] = max(acc[t]['best'], 0.3)

    out = {}
    for t in OA_TARGETS:
        a = acc[t]
        pos, neg, unc, best = (a['pos'], a['neg'], a['unc'], a['best'])
        if not (pos or unc) and g_pos and FEATURES['oa_inherit'] and not touched[t]:
            if neg:
                score, conf = _grade(0, neg, 0, 0.0)
                score = max(score, 0.35)
                conf *= 0.7
            else:
                score, conf = _grade(g_pos, 0, 0, g_best * 0.92)
                conf *= 0.75
        else:
            score, conf = _grade(pos, neg + g_neg, unc, best)
        out[t] = (score, conf, pos, neg)
    return out

def extract(report: str) -> dict:
    cls = clauses(report)
    out = {}
    for tgt in PAIRED:
        s, c, npos, nneg = _score_paired(cls, tgt)
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    for tgt, (s, c, npos, nneg) in _score_oa(cls).items():
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    for tgt in ('Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'):
        if tgt == 'Contusion':
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt), context_penalty=DEGENERATIVE_MARROW, context_bonus=TRAUMA)
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt))
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    if FEATURES['synovitis_backoff'] and out['Synovitis__npos'] == 0 and (out['Synovitis__nneg'] == 0):
        proxy = sum((1 for c in cls if SYNOVIAL_PROXY.search(c) and _polarity(c) == 'positive'))
        eff = out['Effusion']
        prior = 0.3 + 0.3 * max(0.0, (eff - 0.5) / 0.45) + 0.06 * min(proxy, 3)
        out['Synovitis'] = min(0.72, prior)
        out['Synovitis__conf'] = 0.18
    return out
