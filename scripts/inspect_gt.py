import csv
from collections import Counter, defaultdict

by_student = defaultdict(list)
all_labels = []

with open('Anno/ground_truth.csv', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        frame = int(row['frame'])
        sid = int(row['student_id'])
        att = row['attention']
        by_student[sid].append(att)
        all_labels.append(att)

print(f'Total observations: {len(all_labels)}')
print('Overall class distribution:', dict(Counter(all_labels)))
total = len(all_labels)
for k, v in Counter(all_labels).items():
    print(f'  {k}: {v} ({v/total*100:.1f}%)')

print('\nPer-student distribution:')
for sid in sorted(by_student.keys()):
    c = Counter(by_student[sid])
    dominant = c.most_common(1)[0][0]
    att = c.get('Attentive', 0)
    mod = c.get('Moderate', 0)
    non = c.get('Non-Attentive', 0)
    print(f'  Student {sid:2d} (n={len(by_student[sid])}): Attentive={att:3d}, Moderate={mod:3d}, Non-Attentive={non:3d} | Dominant: {dominant}')
