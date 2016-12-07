data = data.strip().split('\n')
header = re.split('\t| ', data[0])
list = []
for word in data[1:]:
    part1 = re.split('\t', word)
    part2 = part1[0].split()
    if '?' in part2:
        continue
    list.append(part2)
final1 = [pandas.to_numeric(x) for x in list]
final2 = numpy.array(final1)
final = pandas.DataFrame(final2, columns=header[0:8])
print(final)