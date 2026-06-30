	from datetime import datetime, timedelta
		import csv
		
		# =========================
		# CONFIGURACOES
		# =========================
		group_id = 5443
		csv_file = "zabbix_data_group_5443_all_items_last_3_months.csv"
		
		# Ultimos 3 meses (~90 dias)
		time_to = int(datetime.now().timestamp())
		time_from = int((datetime.now() - timedelta(days=90)).timestamp())
		
		# Chunk para evitar timeout e respostas gigantes
		CHUNK_HOURS = 6
		CHUNK_SECONDS = CHUNK_HOURS * 3600
		BATCH_ITEMIDS = 200
		
		ONLY_ENABLED_ITEMS = True
		ONLY_NUMERIC = False
		
		csv_columns = ["host_name", "item_name", "timestamp", "value"]
		
		def chunk_list(lst, n):
		"""Divide lista em blocos de tamanho n."""
		for i in range(0, len(lst), n):
		yield lst[i:i+n]
		
		hosts = zapi.host.get({
			"groupids": group_id,
			"output": ["hostid", "name"]
		})
		
		if not hosts:
		print(f"Nenhum host encontrado no grupo {group_id}.")
		raise SystemExit(0)
		
		hostid_to_name = {h["hostid"]: h["name"] for h in hosts}
		hostids = list(hostid_to_name.keys())
		
		items = zapi.item.get({
			"hostids": hostids,
			"output": ["itemid", "name", "hostid", "value_type", "status"]
		})
		
		if ONLY_ENABLED_ITEMS:
		items = [it for it in items if str(it.get("status", "0")) == "0"]
		
		if ONLY_NUMERIC:
		items = [it for it in items if int(it.get("value_type", 999)) in (0, 3)]
		
		item_meta = {}
		items_by_type = {}
		for it in items:
		itemid = it["itemid"]
		host_name = hostid_to_name.get(it["hostid"], it["hostid"])
		item_name = it["name"]
		value_type = int(it["value_type"])
		
		item_meta[itemid] = (host_name, item_name, value_type)
		items_by_type.setdefault(value_type, []).append(itemid)
		
		# EXPORTAR CSV
		with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=csv_columns)
		writer.writeheader()
		
		for value_type, itemids in sorted(items_by_type.items()):
		for batch in chunk_list(itemids, BATCH_ITEMIDS):
		start = time_from
		while start < time_to:
		end = min(start + CHUNK_SECONDS, time_to)
		history = zapi.history.get({
			"history": value_type,
			"itemids": batch,
			"time_from": start,
			"time_till": end,
			"output": ["itemid", "clock", "value"],
			"sortfield": "clock",
			"sortorder": "ASC"
		})
		
		if history:
		for record in history:
		itemid = record["itemid"]
		host_name, item_name, _ = item_meta.get(itemid, ("UNKNOWN", "UNKNOWN", value_type))
		timestamp = datetime.fromtimestamp(int(record["clock"])).strftime("%Y-%m-%d %H:%M:%S")
		
		writer.writerow({
			"host_name": host_name,
			"item_name": item_name,
			"timestamp": timestamp,
			"value": record["value"]
		})
		start = end
		print(f"Exportacao concluida! Arquivo gerado: {csv_file}")