"""Small embedded Arabic word list + random token generators used to build
synthetic training sentences. Not linguistically exhaustive - just enough
lexical variety to bootstrap the model; random letter/diacritic tokens fill
in coverage for rare characters."""

import random

from vocab import ARABIC_LETTERS, DIACRITICS, DIGITS_ARABIC_INDIC, DIGITS_WESTERN

COMMON_WORDS = """
من في على إلى عن مع بعد قبل عند حتى بين خلال ضد لدى
هذا هذه ذلك تلك هؤلاء أولئك الذي التي الذين اللواتي
أنا أنت أنتِ هو هي نحن أنتم أنتن هم هن
كان يكون تكون كانت يكونون أصبح صار ليس ما زال
قال يقول قالت تقول قلت أقول نقول يقولون
ذهب يذهب ذهبت أذهب نذهب يذهبون رجع يرجع
كتب يكتب كتبت أكتب كتاب كتب كاتب مكتوب مكتبة
قرأ يقرأ قراءة قارئ مقروء درس يدرس دراسة مدرسة طالب معلم
عمل يعمل عملت عمال عامل أعمال شغل يشغل وظيفة
بيت منزل دار بناية مدينة قرية شارع طريق ميدان
عالم دنيا أرض سماء شمس قمر نجم بحر نهر جبل واد صحراء
ماء هواء نار تراب حجر رمل شجر زهر ورد
يوم ليل صباح مساء ساعة دقيقة ثانية أسبوع شهر سنة عام
واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة عشرة
مائة ألف مليون أول ثاني ثالث أخير
كبير صغير طويل قصير جميل قبيح جديد قديم سريع بطيء
حار بارد سهل صعب قوي ضعيف غني فقير سعيد حزين
أحمر أزرق أخضر أصفر أسود أبيض بني رمادي
أب أم أخ أخت ابن ابنة جد جدة عم عمة خال خالة زوج زوجة
ولد بنت طفل رجل امرأة إنسان شعب أمة وطن دولة حكومة
جيش شرطة قاضي محكمة قانون حق واجب حرية عدالة سلام حرب
طعام شراب خبز ماء لحم سمك فاكهة خضار حليب سكر ملح
سيارة قطار طائرة سفينة دراجة باب نافذة طاولة كرسي سرير
حاسوب هاتف تلفاز راديو إنترنت برنامج تطبيق شبكة
علم فن رياضة موسيقى تاريخ جغرافيا فيزياء كيمياء رياضيات
دين إسلام مسجد صلاة صوم زكاة حج قرآن نبي رسول
طبيب مستشفى دواء مرض صحة جسم قلب رأس يد رجل عين أذن
حب صداقة سعادة فرح حزن غضب خوف أمل
كل بعض جميع غير نفس مثل كذلك أيضا فقط جدا
نعم لا ربما دائما أبدا أحيانا غدا أمس اليوم الآن
""".split()


def random_word(min_len=2, max_len=8) -> str:
    length = random.randint(min_len, max_len)
    letters = [random.choice(ARABIC_LETTERS) for _ in range(length)]
    if random.random() < 0.5:
        pos = random.randrange(len(letters))
        letters.insert(pos + 1, random.choice(DIACRITICS))
    return "".join(letters)


def random_number() -> str:
    digits = DIGITS_ARABIC_INDIC if random.random() < 0.7 else DIGITS_WESTERN
    length = random.randint(1, 4)
    return "".join(random.choice(digits) for _ in range(length))


def random_token() -> str:
    r = random.random()
    if r < 0.6:
        return random.choice(COMMON_WORDS)
    elif r < 0.85:
        return random_word()
    else:
        return random_number()


def random_sentence(min_tokens=1, max_tokens=8) -> str:
    n = random.randint(min_tokens, max_tokens)
    tokens = [random_token() for _ in range(n)]
    return " ".join(tokens)
