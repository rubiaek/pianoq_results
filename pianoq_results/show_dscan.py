from pianoq.lab.mplc.discrete_scan_result import DiscreetScanResult

path = sys.argv[1]
name = os.path.basename(path)

r = DiscreetScanResult()
r.loadfrom(path)
r.show()

plt.show()
