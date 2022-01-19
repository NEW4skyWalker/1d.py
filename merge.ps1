$i = 1
foreach($dir in gci E:\ethtraffic\tcp\source\){
	mergecap -w E:\Etherum\merge_pcap\ethTcp-$i.pcap E:\Etherum\zzshu_data\tcp\source\$dir\*.pcap
	$i = $i + 1
}

foreach($f in gci E:\ethtraffic\merge_pcap\) {
	mergecap -w E:\Etherum\tcp\ethTcp.pcap E:\Etherum\merge_pcap\*.pcap
}
