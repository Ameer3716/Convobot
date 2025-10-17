# Empathetic Bot — Evaluation Report

_Generated: **2025-10-17 11:00 UTC**_

## 1) Model & Training Config

| Key | Value |
|---|---|
| batch_size | 32 |
| lr | 0.0002 |
| epochs | 5 |
| d_model | 256 |
| nhead | 2 |
| enc_layers | 2 |
| dec_layers | 2 |
| dropout | 0.1 |
| decoding_method | greedy |

## 2) Test Metrics

| Metric | Value |
|---|---|
| Loss | 3.6845 |
| Perplexity | 39.83 |
| BLEU | 0.2316 |
| ROUGE-L | 0.0000 |
| chrF | 0.0000 |

Best validation BLEU: **0.2372**

## 3) Examples (Input → Reference → Prediction)

**Example 1**

- **Input**:  ⁇ motion: grateful |  ⁇ ituation: i am pleased that i got to meet donald trump . |  ⁇ ustomer: i am pleased that i got to meet donald trump . |  ⁇ gent:
- **Reference**: i don't believe you .
- **Prediction**: i am so happy for you .

**Example 2**

- **Input**:  ⁇ motion: confident |  ⁇ ituation: last winters , i went uphill and took a 2 hour skiing classes which really helped to enjoy skiing there after |  ⁇ ustomer: yeah it's an amazing experience altogether . i really enjoy the sport . |  ⁇ gent:
- **Reference**: i do like to watch it in the olympics or sure though !
- **Prediction**: i bet you were able to do that .

**Example 3**

- **Input**:  ⁇ motion: surprised |  ⁇ ituation: i was startled when i walked up on a snake a few days ago in my backyard . |  ⁇ ustomer: i got startled by a snake in my yard a few days ago . |  ⁇ gent:
- **Reference**: i would be the same way . did the snake attack you ?
- **Prediction**: oh no ! what was it ?

**Example 4**

- **Input**:  ⁇ motion: annoyed |  ⁇ ituation: was irked when i saw my dog had gotten out |  ⁇ ustomer: was irked when i saw my dog had gotten out |  ⁇ gent:
- **Reference**: was the dog okay
- **Prediction**: oh no ! did you get it ?

**Example 5**

- **Input**:  ⁇ motion: excited |  ⁇ ituation: i am going to visit my friend in france next month . i cannot wait ! |  ⁇ ustomer: oh wow . . first time going to france ? |  ⁇ gent:
- **Reference**: no . i usually go once or twice a year , but it has been awhile this time , so i am so happy !
- **Prediction**: i am going to be a lot of fun .

**Example 6**

- **Input**:  ⁇ motion: angry |  ⁇ ituation: i had a huge conflict with my parents about moving to another country and i was simply enraged to have them trying to control my life . |  ⁇ ustomer: that's a bummer . what happened once you moved there ? |  ⁇ gent:
- **Reference**: it was an awesome experience and after i came back to the us i felt so fulfilled and my perspective on everything has changed .
- **Prediction**: i was so upset about it . i was so upset about it .

**Example 7**

- **Input**:  ⁇ motion: sad |  ⁇ ituation: i have been struggling with a flare up from my lupus after i had such a long period of feeling good . its depressing to be in bed all day and fail your family in what you need to be doing . |  ⁇ ustomer: i am sorry you have that . it must be rough |  ⁇ gent:
- **Reference**: thank you . yes , chronic illness can be really hard . i try to stay positive because i know it makes a different on my health to stay optimistic . but this flare came suddenly and its horrible to be back in bed .
- **Prediction**: yeah , i have been working on a lot of work .

**Example 8**

- **Input**:  ⁇ motion: sentimental |  ⁇ ituation: my uncle was in the hospital . |  ⁇ ustomer: yea he was , i felt so sentimental that week . i could not even concentrate on my job . |  ⁇ gent:
- **Reference**: i can understand that
- **Prediction**: that's great ! i hope you find a good job .

**Example 9**

- **Input**:  ⁇ motion: anxious |  ⁇ ituation: uni is starting up soon and im nervous |  ⁇ ustomer: uni is starting up soon and im nervous |  ⁇ gent:
- **Reference**: why are you nervous ?
- **Prediction**: i'm sure you will do well .

**Example 10**

- **Input**:  ⁇ motion: apprehensive |  ⁇ ituation: i am weary about ordering chicken now . it made me really sick once |  ⁇ ustomer: i don't order chicken anymore from restaurants |  ⁇ gent:
- **Reference**: why not ? i love chicken !
- **Prediction**: oh , what kind of restaurant ?

**Example 11**

- **Input**:  ⁇ motion: surprised |  ⁇ ituation: i was so surprized on the night of my graduation my parents bought me a new car ! |  ⁇ ustomer: a saturn |  ⁇ gent:
- **Reference**: i remember those cars ! how long did it last you ?
- **Prediction**: what kind of car was it ?

**Example 12**

- **Input**:  ⁇ motion: afraid |  ⁇ ituation: my husband went out of town for a few days for work . every sound i hear makes my hair stand up on end , i may not survive this trip . |  ⁇ ustomer: my husband went out of town for a few days . every sound i hear makes my hair stand up on end ! |  ⁇ gent:
- **Reference**: oh no . how long will he be away for ?
- **Prediction**: oh no ! i'm sorry to hear that .

**Example 13**

- **Input**:  ⁇ motion: faithful |  ⁇ ituation: i love my wife so much . i would never cheat on her . |  ⁇ ustomer: that's nice to hear ! have you treated her to something special ? |  ⁇ gent:
- **Reference**: i always do . i would never cheat on her .
- **Prediction**: i am not sure . i am so happy for her .

**Example 14**

- **Input**:  ⁇ motion: ashamed |  ⁇ ituation: when my mom started yelling at someone in the store for no reason . |  ⁇ ustomer: that must be very challenging to deal with at times . |  ⁇ gent:
- **Reference**: it is very challenging to deal with , along with being very embarrassing . sometimes i am ashamed of her .
- **Prediction**: yeah , i can't even know how to do that .

**Example 15**

- **Input**:  ⁇ motion: nostalgic |  ⁇ ituation: i found my nintendo in the closet a few months . i remembered all the games i used ot play on it . |  ⁇ ustomer: i found my nintendo entertainment system in my closet a few months ago |  ⁇ gent:
- **Reference**: was it a nice surprise ?
- **Prediction**: that's awesome ! i'm sure you'll do that !

**Example 16**

- **Input**:  ⁇ motion: sentimental |  ⁇ ituation: i found an old stuffed animal in my closet . it made me yearn for earlier times . |  ⁇ ustomer: yeah . a fluffy one . i made me think back to when i had a pet koala . |  ⁇ gent:
- **Reference**: i bet that was pretty awesome . how'd you grow your eucalyptus ?
- **Prediction**: that's good . i'm sure you'll find a good one .

**Example 17**

- **Input**:  ⁇ motion: jealous |  ⁇ ituation: my dad got my brother a watch but not me . |  ⁇ ustomer: did you want a watch ? |  ⁇ gent:
- **Reference**: yes , and i feel kind of envious of my brother now .
- **Prediction**: i did , he was a good one .

**Example 18**

- **Input**:  ⁇ motion: anxious |  ⁇ ituation: i am so worried about my math test coming up . i am afraid i will flunk it . i need to study more . |  ⁇ ustomer: don't think that well ! as long as you are prepared , you will do great ! |  ⁇ gent:
- **Reference**: yeah i think i need to study some more . i hate trigonometry so much .
- **Prediction**: i am so nervous about it . i am so nervous about it .

**Example 19**

- **Input**:  ⁇ motion: hopeful |  ⁇ ituation: i got an interview for a nannying job . it is full-time for two twin babies . |  ⁇ ustomer: i just got a job interview ! |  ⁇ gent:
- **Reference**: you must be so happy ? what was the job you got a interview for ?
- **Prediction**: that's great ! i hope you get it !

**Example 20**

- **Input**:  ⁇ motion: disappointed |  ⁇ ituation: my friend recommended me a restaurant . i went with my wife to try it . the food was terrible . |  ⁇ ustomer: my friend recommended me a restaurant . i went with my wife to try it . the food was terrible . |  ⁇ gent:
- **Reference**: maybe you should go by yourself . normally wifes stay at home and clean .
- **Prediction**: oh no ! that's so sad .

**Example 21**

- **Input**:  ⁇ motion: nostalgic |  ⁇ ituation: i have a notebook of all my daughters scribbles and drawings |  ⁇ ustomer: i am . she is left handed so i think she will be very artistic |  ⁇ gent:
- **Reference**: maybe she will eventually want to study art .
- **Prediction**: i'm sorry to hear that . i'm sure she will be a good time .

**Example 22**

- **Input**:  ⁇ motion: grateful |  ⁇ ituation: my father paid for my university tuition . he allows me to save lots of money . |  ⁇ ustomer: he' |  ⁇ gent:
- **Reference**: i wish everyone was that lucky .
- **Prediction**: he is a great job .

**Example 23**

- **Input**:  ⁇ motion: disgusted |  ⁇ ituation: that the neighbor lets her cats use the bathroom outside my window |  ⁇ ustomer: i have but she doesnt seem to mind at all . it just smells gross . |  ⁇ gent:
- **Reference**: that's really inconsiderate of her .
- **Prediction**: i'm sorry to hear that .

**Example 24**

- **Input**:  ⁇ motion: surprised |  ⁇ ituation: my son just said his first words this weekend . they just came out of nowhere . |  ⁇ ustomer: it was ! he has been a little bit of a late bloomer , but it was worth the wait . |  ⁇ gent:
- **Reference**: for sure . before you know it , he'll be saying full sentences !
- **Prediction**: i'm sure he'll be able to get it .

**Example 25**

- **Input**:  ⁇ motion: apprehensive |  ⁇ ituation: there was a time when i finally dredged up the courage to visit my neighbor's chickens . i was really scared that they would peck at me , because i've heard that the roosters are very angsty and territorial . |  ⁇ ustomer: there were around 20 at the time . however , i soon learned that they didn't want to eat me . |  ⁇ gent:
- **Reference**: oh good . well what did they end up wanting ?
- **Prediction**: that's a good idea . i'm glad you were able to get a lot of money .

---
_Full predictions CSV: `report/test_predictions.csv`_
_Sampled examples CSV: `report/evaluation_examples.csv`_