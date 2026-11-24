// Restaurant staff drafts are kept on this browser, separately for each table and screen.
function orderDesk(user, screen) {
    const prefix = 'neodine:' + user + ':' + screen + ':';
    function read(key) {
        try { return JSON.parse(localStorage.getItem(prefix + key)); }
        catch { return null; }
    }
    function write(key, value) {
        if (value === null) localStorage.removeItem(prefix + key);
        else localStorage.setItem(prefix + key, JSON.stringify(value));
    }
    function confirmKitchen(result) {
        const text = 'Table ' + result.table_num + ' · ' + result.quantity +
            ' item(s) received by kitchen · Order #' + result.order_id;
        write('confirmation', text);
        return text;
    }
    async function send(table, items, note) {
        const key = 'sending:' + table;
        const previous = read(key);
        const body = previous || { submission: crypto.randomUUID(), table_num: table, items, note };
        write(key, body);
        let response;
        if (previous) {
            response = await fetch('/api/kitchen/submissions/' + body.submission);
            if (response.status !== 404 && !response.ok) throw new Error('Could not verify kitchen receipt. Retry to check again.');
        }
        if (!response || response.status === 404) {
            response = await fetch('/waiter/place-order', {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
            });
        }
        const result = await response.json();
        if (!response.ok || !result.success) {
            if (response.status >= 400 && response.status < 500) write(key, null);
            throw new Error(result.error || 'Kitchen receipt unconfirmed. Retry to check.');
        }
        return {text: confirmKitchen(result), items: body.items, note: body.note};
    }
    return {read, write, send};
}
