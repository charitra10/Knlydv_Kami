#include "helper.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace bot {
using namespace unswbc;

constexpr int MAX_CELLS = 4096, INF = 100000;
constexpr int BFS_Q = MAX_CELLS;
constexpr int SONAR_TTL = 36, SONAR_ID_MASK = 8191;
const auto DIRS = Direction::get_direction_list();

inline int di(Direction d) {
    for (int i = 0; i < 4; ++i)
        if (DIRS[i] == d)
            return i;
    return 0;
}

enum class Mode { Disperse, Forage, Ambush, Escape, Portal, Reside, Return, Feed, Split, Kill, Trapped };

inline const char *name(Mode m) {
    static const char *names[] = {"disperse", "forage", "ambush", "escape", "portal", "reside",
                                  "return",   "feed",   "split",  "kill",   "trapped"};
    return names[static_cast<int>(m)];
}

struct Action {
    std::vector<Direction> moves;
    int child = 0;
    Mode mode = Mode::Forage;
    bool intentional_death = false;
};

struct Cell {
    int seen = -1, visited = -10000, pearl_round = -1;
    bool has_dragon = false;
    std::array<int, 4> edge{{-2, -2, -2, -2}}; // unknown=-2, kelp=-1, open=0, portal=id+1
};

struct Portal {
    int id;
    std::vector<std::pair<Position, int>> ends;
    int occupied_until = -1;
    bool occupied = false;
};

struct AlphaTrack {
    int id, seen, len = 8;
    Position p;
};

struct DragonState {
    bool initialized = false, alpha = false, growing = false, resident = false, dispersing = false, evacuating = false;
    bool kamikaze_lock = false, tail_visible = false;
    int born = 0, last_food = 0, sector = 0, home_portal = -1, pending_portal = -1, explore_timer = 0;
    int kamikaze_signal_round = -1000, kamikaze_lock_until = -1000, harvester_until = -1000, last_combat_split = -1000, tail_facing = -1;
    Position sector_target;
    Position explore_target;
    Position return_tile;
    Position harvest_anchor{0, 0};
    Position tail_pos{0, 0};
    Position enemy_alpha_pos{0, 0};
    int enemy_alpha_seen = -1000;
    // Rolling memory of our own recent trajectory — used to escape tight oscillation loops
    std::vector<Position> recent_path;
    std::array<Cell, MAX_CELLS> cells{};
    std::vector<Portal> portals;
    std::vector<AlphaTrack> alphas;
    std::vector<DragonPart> previous_heads;
};

class Brain {
    Controller &c;
    Game &g;
    int w, h, round = 0;
    Position here;
    std::vector<DragonPart> friends, enemies;
    bool sonar_enemy_alert = false;

    std::array<int, MAX_CELLS> distance{}, first{}, predecessor{}, arrival{};
    mutable std::array<int, MAX_CELLS> risk_cache{};
    mutable std::array<int, MAX_CELLS> danger_depth{};
    mutable std::array<bool, MAX_CELLS> space_seen{};
    mutable std::array<int, MAX_CELLS> remembered_first{};
    mutable std::array<Position, MAX_CELLS> remembered_q{};

    std::vector<std::pair<int, int>> friend_len_cache, enemy_len_cache;
    std::vector<int> friend_alpha_cache, enemy_edge_cache;

  public:
    DragonState s;

    Brain(Controller &controller, Game &game_state) : c(controller), g(game_state) {
        std::tie(w, h) = g.get_map_size();
        danger_depth.fill(-1);
        space_seen.fill(false);
        remembered_first.fill(-1);
    }

    int index(Position p) const { return p.y * w + p.x; }

    Position wrap(Position p) const {
        return {(p.x % w + w) % w, (p.y % h + h) % h};
    }

    Position step(Position p, int d) const {
        auto [x, y] = DIRS[d].get_offset();
        return wrap({p.x + x, p.y + y});
    }

    int delta(int a, int b, int size) const {
        int v = (b - a + size) % size;
        if (v > size / 2) v -= size;
        return v;
    }

    int dist(Position a, Position b) const {
        return std::abs(delta(a.x, b.x, w)) + std::abs(delta(a.y, b.y, h));
    }

    int chebyshev(Position a, Position b) const {
        return std::max(std::abs(delta(a.x, b.x, w)), std::abs(delta(a.y, b.y, h)));
    }

    Portal &portal(int id) {
        for (auto &p : s.portals)
            if (p.id == id) return p;
        s.portals.push_back({id, {}, -1, false});
        return s.portals.back();
    }

    std::pair<Position, int> canonical(Position p, int d) const {
        if (d == 2) return {step(p, 2), 0};
        if (d == 1) return {step(p, 1), 3};
        return {p, d};
    }

    std::optional<Position> destination(Position p, int d) const {
        int e = s.cells[index(p)].edge[d];
        if (e < 0) return {};
        if (e == 0) return step(p, d);
        auto from = canonical(p, d);
        for (const auto &port : s.portals)
            if (port.id == e - 1 && port.ends.size() == 2) {
                if (port.ends[0] != from && port.ends[1] != from) return {};
                auto to = port.ends[0] == from ? port.ends[1] : port.ends[0];
                return (d == 0 || d == 3) ? step(to.first, d) : to.first;
            }
        return {};
    }

    bool empty(Position p) const {
        const auto &cell = s.cells[index(p)];
        return cell.seen == round && !cell.has_dragon;
    }

    void remember_alpha(int id, Position p, int origin_round, int len = 8, bool exact_len = false) {
        for (auto &a : s.alphas)
            if (a.id == id) {
                if (origin_round >= a.seen) {
                    int updated_len = exact_len ? len : ((origin_round > a.seen + 6) ? len : std::max(len, a.len));
                    a = {id, origin_round, updated_len, p};
                }
                return;
            }
        s.alphas.push_back({id, origin_round, len, p});
    }

    int friendly_visible_length(int friend_id) const {
        for (const auto &p : friend_len_cache)
            if (p.first == friend_id) return p.second;
        return 0;
    }

    int enemy_visible_length(int enemy_id) const {
        for (const auto &p : enemy_len_cache)
            if (p.first == enemy_id) return p.second;
        return 0;
    }

    int enemy_effective_length(int enemy_id) const {
        int vis = enemy_visible_length(enemy_id);
        for (int eid : enemy_edge_cache)
            if (eid == enemy_id) return vis + 1;
        return vis;
    }

    bool has_longer_visible_enemy() const {
        if (s.kamikaze_lock && round <= s.kamikaze_lock_until) return !enemies.empty();
        if (c.get_unit_count() < 4) return false;
        int my_len = c.get_length();
        for (const auto &e : enemies) {
            int elen = enemy_effective_length(e.get_id());
            if (elen > my_len) return true;
        }
        return false;
    }

    int team_zero_id_val() const {
        return (c.get_team() == Team::A) ? 0 : 1;
    }

    bool is_primary_alpha_id(int id) const {
        return id == team_zero_id_val();
    }

    bool is_alpha(int id) const {
        if (id == c.get_id()) return s.alpha;
        for (int aid : friend_alpha_cache)
            if (aid == id) return true;
        return false;
    }

    bool is_superior_alpha_track(const AlphaTrack &a) const {
        int my_id = (c.get_id() & SONAR_ID_MASK);
        if (a.id == my_id || round - a.seen > SONAR_TTL) return false;
        int my_len = c.get_length();
        if (a.len > my_len) return true;
        if (a.len == my_len)
            return is_primary_alpha_id(a.id) || (!is_primary_alpha_id(c.get_id()) && a.id < my_id);
        return false;
    }

    std::optional<AlphaTrack> get_superior_alpha() const {
        std::optional<AlphaTrack> best;
        for (const auto &a : s.alphas) {
            if (!is_superior_alpha_track(a)) continue;
            if (!best || a.len > best->len || (a.len == best->len && dist(here, a.p) < dist(here, best->p)))
                best = a;
        }
        return best;
    }

    int map_scale() const {
        return static_cast<int>(std::sqrt(w * h));
    }

    int kamikaze_threshold() const {
        int n = map_scale();
        if (n <= 12) return 5;
        if (n < 20)  return 10;
        if (n <= 26) return 18;
        if (n <= 31) return 28;
        if (n <= 35) return 18;
        if (n <= 50) return 26;
        return 48;
    }

    int alpha_split_cap() const {
        int n = map_scale();
        return n <= 12 ? 42 : n < 20 ? 36 : n <= 26 ? 24 : n <= 35 ? 18 : n <= 50 ? 12 : 8;
    }

    bool is_adaptive_kamikaze_active() const {
        return round < 415 && (round - s.kamikaze_signal_round <= 28) && c.get_unit_count() >= 6;
    }

    bool is_kamikaze_regime() const {
        return round < 415 && (c.get_unit_count() > kamikaze_threshold() || is_adaptive_kamikaze_active());
    }

    bool is_kamikaze() const {
        if (s.kamikaze_lock && round <= s.kamikaze_lock_until) return true;
        return !s.alpha && !s.growing && is_kamikaze_regime() && c.get_length() == 2;
    }

    bool is_sprint_hunter() const {
        if (s.kamikaze_lock && round <= s.kamikaze_lock_until && c.get_length() >= 3) return true;
        return !s.alpha && !s.growing && is_kamikaze_regime() && c.get_length() == 3;
    }

    void occupy(int id) {
        auto &p = portal(id);
        p.occupied = true;
        p.occupied_until = INF;
    }

    int center_dist(Position p) const {
        return std::abs(p.x - w / 2) + std::abs(p.y - h / 2);
    }

    Position sector_waypoint(int sec) const {
        int cx = w / 2, cy = h / 2;
        bool large_map = (map_scale() > 35);
        int ring = (c.get_id() / 2 + sec) % 3;
        int rx = (ring == 0 ? std::max(2, w / 7) : (ring == 1 ? std::max(3, w / 4) : std::max(4, (w * 3) / 8)));
        int ry = (ring == 0 ? std::max(2, h / 7) : (ring == 1 ? std::max(3, h / 4) : std::max(4, (h * 3) / 8)));
        int d_rx = large_map ? std::max(2, (rx * 3) / 4) : rx;
        int d_ry = large_map ? std::max(2, (ry * 3) / 4) : ry;
        int jitter_x = (static_cast<int>((c.get_id() * 3 + 1) % 3)) - 1;
        int jitter_y = (static_cast<int>((c.get_id() * 5 + 2) % 3)) - 1;
        Position base;
        switch ((sec % 8 + 8) % 8) {
        case 0: base = {cx, cy - ry}; break;
        case 1: base = {cx, cy + ry}; break;
        case 2: base = {cx - rx, cy}; break;
        case 3: base = {cx + rx, cy}; break;
        case 4: base = {cx - d_rx, cy - d_ry}; break;
        case 5: base = {cx + d_rx, cy + d_ry}; break;
        case 6: base = {cx + d_rx, cy - d_ry}; break;
        default: base = {cx - d_rx, cy + d_ry}; break;
        }
        return {std::clamp(base.x + jitter_x, 2, std::max(2, w - 3)),
                std::clamp(base.y + jitter_y, 2, std::max(2, h - 3))};
    }

    int threshold() const {
        int n = map_scale();
        return std::clamp(480 - 6 * n, 120, 410);
    }

    std::uint64_t alpha_packet64(int id, Position p, int origin_round, int alpha_len) const {
        std::uint64_t sig = (c.get_team() == Team::A) ? 1ULL : 2ULL;
        std::uint64_t aid = static_cast<std::uint64_t>(id & SONAR_ID_MASK);
        std::uint64_t r = static_cast<std::uint64_t>(std::clamp(origin_round, 0, 511));
        std::uint64_t px = static_cast<std::uint64_t>(p.x & 63);
        std::uint64_t py = static_cast<std::uint64_t>(p.y & 63);
        std::uint64_t alen = static_cast<std::uint64_t>(std::clamp(alpha_len, 0, 255));
        bool has_e = (round - s.enemy_alpha_seen <= 12);
        std::uint64_t he = has_e ? 1ULL : 0ULL;
        std::uint64_t ex = has_e ? static_cast<std::uint64_t>(s.enemy_alpha_pos.x & 63) : 0ULL;
        std::uint64_t ey = has_e ? static_cast<std::uint64_t>(s.enemy_alpha_pos.y & 63) : 0ULL;
        std::uint64_t eage = has_e ? static_cast<std::uint64_t>(std::clamp(round - s.enemy_alpha_seen, 0, 15)) : 0ULL;
        std::uint64_t kam_bucket = 0ULL;
        if (is_kamikaze_regime()) {
            int age = (c.get_unit_count() > kamikaze_threshold()) ? 0 : std::clamp(round - s.kamikaze_signal_round, 0, 27);
            kam_bucket = static_cast<std::uint64_t>(std::clamp(age / 4, 0, 6) + 1);
        }
        return (sig << 62) | (aid << 49) | (r << 40) | (px << 34) | (py << 28) |
               (alen << 20) | (he << 19) | (ex << 13) | (ey << 7) | (kam_bucket << 4) | eage;
    }

    bool enclosed_nursery() const {
        int lo = 0, hi = 0;
        bool has_portal = false;
        remembered_q[hi++] = here;
        remembered_first[index(here)] = 0;
        auto cleanup = [&]() {
            for (int i = 0; i < hi; ++i) remembered_first[index(remembered_q[i])] = -1;
        };
        while (lo < hi) {
            Position p = remembered_q[lo++];
            if (hi > std::min(49, w * h / 4)) { cleanup(); return false; }
            for (int d = 0; d < 4; ++d) {
                int e = s.cells[index(p)].edge[d];
                if (e == -2) { cleanup(); return false; }
                if (e > 0) has_portal = true;
                if (e != 0) continue;
                auto n = step(p, d);
                if (remembered_first[index(n)] >= 0) continue;
                remembered_first[index(n)] = 0;
                remembered_q[hi++] = n;
            }
        }
        cleanup();
        return has_portal;
    }

    void check_adaptive_kamikaze() {
        if (round >= 415) return;
        if (round - s.born <= 1) {
            int my_units = c.get_unit_count();
            int enemy_max = std::max(0, c.get_id() - my_units);
            if (my_units >= 7 && (my_units - enemy_max >= 5 || (my_units >= 10 && my_units * 2 >= enemy_max * 3))) {
                s.kamikaze_signal_round = round;
            }
        }
        if (c.get_unit_count() > kamikaze_threshold()) {
            s.kamikaze_signal_round = round;
        }
    }

    void initialize() {
        s.initialized = true;
        s.born = round;
        s.last_food = round;

        s.alpha = is_primary_alpha_id(c.get_id());
        if (round > 1) {
            if (c.get_length() <= 3 && !enemies.empty()) {
                for (const auto &e : enemies) {
                    if (enemy_effective_length(e.get_id()) > c.get_length() || e.get_id() <= 1) {
                        s.kamikaze_lock = true;
                        s.kamikaze_lock_until = round + 8;
                        break;
                    }
                }
            } else if (c.get_length() >= 4) {
                s.harvester_until = round + 10;
                if (c.get_length() >= 6) s.alpha = true;
            }
        }
        s.evacuating = !s.alpha && c.get_id() > 1 && !s.kamikaze_lock && enclosed_nursery();

        s.sector = (c.get_id() / 2) % 8;
        s.sector_target = sector_waypoint(s.sector);
        s.explore_timer = 0;
        s.dispersing = true;
        check_adaptive_kamikaze();
    }

    void observe() {
        round = g.get_round_num();
        here = c.get_position();
        friends.clear();
        enemies.clear();
        risk_cache.fill(-1);

        auto echoes = c.get_sonar_echoes();
        sonar_enemy_alert = (echoes.enemy_head > 0 || echoes.enemy >= 2);

        auto cur_tile = c.get_tile(here);
        if (cur_tile && cur_tile->has_pearl()) s.last_food = round;

        friend_len_cache.clear();
        enemy_len_cache.clear();
        enemy_edge_cache.clear();
        std::vector<DragonPart> my_parts;

        for (const auto &t : c.get_tiles()) {
            auto p = t.get_position();
            auto &cell = s.cells[index(p)];
            cell.seen = round;
            auto part = t.get_dragon();
            cell.has_dragon = (part != nullptr);
            if (t.has_pearl() && !part) {
                cell.pearl_round = round;
            } else if (!t.has_pearl() && t.get_pearl_time() > 0 && t.get_pearl_time() <= 14 && !part) {
                cell.pearl_round = round + t.get_pearl_time();
            } else {
                cell.pearl_round = -1;
            }
            for (int d = 0; d < 4; ++d) {
                const auto &e = t.get_edge(DIRS[d]);
                cell.edge[d] = !e.is_passable() ? -1 : e.is_portal() ? e.get_portal_id() + 1 : 0;
                if (e.is_portal()) {
                    auto &port = portal(e.get_portal_id());
                    auto endpoint = canonical(p, d);
                    if (std::find(port.ends.begin(), port.ends.end(), endpoint) == port.ends.end())
                        port.ends.push_back(endpoint);
                }
            }
            if (part) {
                if (part->get_id() == c.get_id()) {
                    my_parts.push_back(*part);
                }
                auto &cache = (part->get_team() == c.get_team()) ? friend_len_cache : enemy_len_cache;
                int id = part->get_id();
                bool found = false;
                for (auto &entry : cache)
                    if (entry.first == id) { ++entry.second; found = true; break; }
                if (!found) cache.push_back({id, 1});
                if (part->is_head() && id != c.get_id())
                    (part->get_team() == c.get_team() ? friends : enemies).push_back(*part);
            }
        }

        for (const auto &t : c.get_tiles()) {
            auto part = t.get_dragon();
            if (part && part->get_team() != c.get_team() && chebyshev(here, t.get_position()) == 3) {
                int id = part->get_id();
                if (enemy_visible_length(id) >= 3 &&
                    std::find(enemy_edge_cache.begin(), enemy_edge_cache.end(), id) == enemy_edge_cache.end()) {
                    enemy_edge_cache.push_back(id);
                }
            }
        }

        s.tail_visible = false;
        if (static_cast<int>(my_parts.size()) == c.get_length()) {
            for (const auto &seg : my_parts) {
                if (seg.is_head()) continue;
                bool pointed_to = false;
                for (const auto &other : my_parts) {
                    if (other.position == seg.position) continue;
                    if (!other.is_head() && step(other.position, di(other.get_dir())) == seg.position) {
                        pointed_to = true;
                        break;
                    }
                }
                if (!pointed_to) {
                    s.tail_visible = true;
                    s.tail_pos = seg.position;
                    s.tail_facing = di(seg.get_dir());
                    break;
                }
            }
        }

        for (const auto &f : friends) s.cells[index(f.position)].pearl_round = -1;
        for (const auto &e : enemies) s.cells[index(e.position)].pearl_round = -1;
        s.cells[index(here)].pearl_round = -1;

        auto by_id = [](const DragonPart &a, const DragonPart &b) { return a.get_id() < b.get_id(); };
        std::sort(friends.begin(), friends.end(), by_id);
        std::sort(enemies.begin(), enemies.end(), by_id);

        if (!s.initialized) initialize();
        if (s.kamikaze_lock && round > s.kamikaze_lock_until) s.kamikaze_lock = false;
        check_adaptive_kamikaze();

        // Enemy Alpha tracking from direct vision
        for (const auto &e : enemies) {
            if (e.get_id() <= 1 || enemy_visible_length(e.get_id()) >= 6) {
                s.enemy_alpha_pos = e.position;
                s.enemy_alpha_seen = round;
            }
        }

        // Decode 64-bit Protocol 3 sonar mesh packets
        std::uint64_t team_sig = (c.get_team() == Team::A) ? 1ULL : 2ULL;
        for (auto msg : c.get_sonar_messages()) {
            if ((msg >> 62) != team_sig) continue;
            int kam_bucket = static_cast<int>((msg >> 4) & 7ULL);
            if (kam_bucket > 0 && round < 415) {
                int est_origin = round - (kam_bucket - 1) * 4;
                if (est_origin <= round && round - est_origin <= 28 && est_origin > s.kamikaze_signal_round) {
                    s.kamikaze_signal_round = est_origin;
                }
            }
            if ((msg >> 19) & 1ULL) {
                int ex = static_cast<int>((msg >> 13) & 63ULL);
                int ey = static_cast<int>((msg >> 7) & 63ULL);
                int eage = static_cast<int>(msg & 15ULL);
                int eseen = round - eage;
                if (ex < w && ey < h && eseen <= round && eseen > s.enemy_alpha_seen) {
                    s.enemy_alpha_pos = {ex, ey};
                    s.enemy_alpha_seen = eseen;
                }
            }
            int alpha_id = static_cast<int>((msg >> 49) & SONAR_ID_MASK);
            if (alpha_id >= SONAR_ID_MASK || alpha_id == (c.get_id() & SONAR_ID_MASK)) continue;
            int origin_round = static_cast<int>((msg >> 40) & 511ULL);
            int age = round - origin_round;
            if (age < 0 || age > SONAR_TTL) continue;
            Position p{static_cast<int>((msg >> 34) & 63ULL), static_cast<int>((msg >> 28) & 63ULL)};
            if (p.x >= w || p.y >= h) continue;
            int alen = static_cast<int>((msg >> 20) & 255ULL);
            remember_alpha(alpha_id, p, origin_round, std::max(3, alen), true);
        }

        // Remember visible alphas directly
        for (auto f : friends) {
            int id = f.get_id();
            int flen = friendly_visible_length(id);
            bool is_a = is_primary_alpha_id(id) || flen > 7;
            if (!is_a) {
                for (const auto &a : s.alphas)
                    if ((a.id & SONAR_ID_MASK) == (id & SONAR_ID_MASK) && round - a.seen <= SONAR_TTL) {
                        is_a = true;
                        break;
                    }
            }
            if (is_a) remember_alpha(id & SONAR_ID_MASK, f.position, round, std::max(3, flen), false);

            int e = s.cells[index(f.position)].edge[(di(f.get_dir()) + 2) % 4];
            if (e > 0 && round > s.born) {
                bool changed = true;
                for (auto prev : s.previous_heads)
                    if (prev.get_id() == f.get_id() && prev.position == f.position) changed = false;
                if (changed) occupy(e - 1);
            }
        }

        s.previous_heads = friends;
        s.alphas.erase(std::remove_if(s.alphas.begin(), s.alphas.end(),
                                      [&](const AlphaTrack &a) { return round - a.seen > SONAR_TTL; }),
                       s.alphas.end());

        // Promote long survivors (>7) to Alpha and originate their own sonar.
        if (!s.alpha && !s.kamikaze_lock && c.get_length() > 7) {
            s.alpha = true;
            s.growing = true;
        }

        friend_alpha_cache.clear();
        for (auto f : friends) {
            int id = f.get_id();
            bool is_a = is_primary_alpha_id(id) || friendly_visible_length(id) > 7;
            if (!is_a) {
                for (const auto &a : s.alphas)
                    if ((a.id & SONAR_ID_MASK) == (id & SONAR_ID_MASK) && round - a.seen <= SONAR_TTL) {
                        is_a = true;
                        break;
                    }
            }
            if (is_a) friend_alpha_cache.push_back(id);
        }

        ++s.explore_timer;
        if (dist(here, s.sector_target) <= 3 || s.explore_timer > std::clamp((w + h) / 2, 16, 64)) {
            s.dispersing = false;
            s.sector = (s.sector + 3) % 8;
            s.sector_target = sector_waypoint(s.sector);
            s.explore_timer = 0;
        }

        int repeats = static_cast<int>(std::count(s.recent_path.begin(), s.recent_path.end(), here));
        if (repeats >= 2 && !s.resident) {
            s.sector = (s.sector + 3) % 8;
            s.sector_target = sector_waypoint(s.sector);
            s.explore_timer = 0;
        }
        s.cells[index(here)].visited = round;

        s.recent_path.push_back(here);
        if (s.recent_path.size() > 24) s.recent_path.erase(s.recent_path.begin());

        if (s.pending_portal >= 0) {
            occupy(s.pending_portal);
            s.resident = !s.evacuating;
            s.home_portal = s.resident ? s.pending_portal : -1;
            s.return_tile = here;
            s.evacuating = false;
            s.last_food = round;
            s.pending_portal = -1;
        }

        if (s.kamikaze_lock) {
            s.growing = false;
        } else if (round <= s.harvester_until && (!enemies.empty() || round <= s.born + 3)) {
            s.growing = true;
        } else if (s.alpha) {
            s.growing = (round > threshold() || c.get_unit_count() >= alpha_split_cap());
        } else {
            s.growing = (round >= 380 && c.get_unit_count() >= 14);
        }
    }

    bool portal_occupied(int id) const {
        for (const auto &p : s.portals)
            if (p.id == id) return p.occupied;
        return false;
    }

    void paths() {
        int n_cells = w * h;
        std::fill_n(distance.begin(), n_cells, INF);
        std::fill_n(first.begin(), n_cells, -1);
        std::fill_n(predecessor.begin(), n_cells, -1);
        std::fill_n(arrival.begin(), n_cells, -1);
        static std::array<Position, BFS_Q> q;
        int lo = 0, hi = 0;
        q[hi++] = here;
        distance[index(here)] = 0;
        while (lo < hi) {
            auto p = q[lo++];
            int pi = index(p);
            for (int d = 0; d < 4; ++d) {
                if (pi == index(here) && d == (di(c.get_dir()) + 2) % 4) continue;
                if (s.cells[pi].edge[d] != 0) continue;
                auto n = step(p, d);
                int ni = index(n);
                auto t = c.get_tile(n);
                if (!t || distance[ni] != INF) continue;
                auto part = t->get_dragon();
                if (part && !(part->get_team() != c.get_team() && part->is_head())) continue;
                distance[ni] = distance[pi] + 1;
                first[ni] = pi == index(here) ? d : first[pi];
                predecessor[ni] = pi;
                arrival[ni] = d;
                if (!part && hi < BFS_Q) q[hi++] = n;
            }
        }
    }

    int escape_count(Position p) const {
        int n = 0;
        for (int d = 0; d < 4; ++d) {
            int edge = s.cells[index(p)].edge[d];
            if (edge == 0 && empty(step(p, d))) ++n;
            else if (edge > 0 && !portal_occupied(edge - 1)) ++n;
        }
        return n;
    }

    int forward_escape_count(Position p, int arr_d) const {
        int back_d = (arr_d >= 0) ? ((arr_d + 2) % 4) : -1;
        int n = 0;
        for (int d = 0; d < 4; ++d) {
            if (d == back_d) continue;
            int edge = s.cells[index(p)].edge[d];
            if (edge == 0) {
                Position nxt = step(p, d);
                if (empty(nxt) || s.cells[index(nxt)].seen < round) ++n;
            } else if (edge > 0 && (!portal_occupied(edge - 1) || s.evacuating)) {
                ++n;
            }
        }
        return n;
    }

    bool is_dead_end_trap(Position p, int arr_d) const {
        if (arr_d < 0) return false;
        int back_d = (arr_d + 2) % 4;
        Position pred = step(p, back_d);
        if (forward_escape_count(p, arr_d) == 0) return true;

        static std::array<Position, 16> q;
        static std::array<int, 16> depth;
        int lo = 0, hi = 0;
        q[hi] = p;
        depth[hi++] = 0;
        space_seen[index(p)] = true;

        auto cleanup = [&]() {
            for (int i = 0; i < hi; ++i) space_seen[index(q[i])] = false;
        };

        while (lo < hi) {
            Position cur = q[lo];
            int dep = depth[lo++];
            if (hi >= 5) { cleanup(); return false; }
            for (int d = 0; d < 4; ++d) {
                if (cur == p && d == back_d) continue;
                int edge = s.cells[index(cur)].edge[d];
                if (edge > 0 && (!portal_occupied(edge - 1) || s.evacuating)) {
                    cleanup();
                    return false;
                }
                if (edge != 0) continue;
                Position nxt = step(cur, d);
                if (nxt == pred && dep >= 2) {
                    cleanup();
                    return false;
                }
                if (s.cells[index(nxt)].seen < round) {
                    cleanup();
                    return false;
                }
                if (!empty(nxt) || space_seen[index(nxt)]) continue;
                space_seen[index(nxt)] = true;
                q[hi] = nxt;
                depth[hi++] = dep + 1;
                if (hi >= 5) { cleanup(); return false; }
            }
        }
        cleanup();
        return true;
    }

    bool is_enemy_certain_death(Position n, int arr_d) const {
        if (enemies.empty()) return false;
        int f_exits = forward_escape_count(n, arr_d);
        if (f_exits == 0) return true;

        for (const auto &e : enemies) {
            int e_opp = (di(e.get_dir()) + 2) % 4;
            int enemy_escapes = 0;
            Position forced_tile = e.position;
            for (int ed = 0; ed < 4; ++ed) {
                if (ed == e_opp) continue;
                int edge = s.cells[index(e.position)].edge[ed];
                if (edge < 0) continue;
                if (edge == 0) {
                    Position ep = step(e.position, ed);
                    if (ep == here || empty(ep)) { ++enemy_escapes; forced_tile = ep; }
                } else ++enemy_escapes;
            }
            if (enemy_escapes == 1 && forced_tile == n) return true;

            if (f_exits == 1 && dist(e.position, n) == 1) {
                int back_d = (arr_d >= 0) ? ((arr_d + 2) % 4) : -1;
                for (int d = 0; d < 4; ++d) {
                    if (d == back_d || s.cells[index(n)].edge[d] != 0) continue;
                    Position sole = step(n, d);
                    if (empty(sole) && dist(e.position, sole) <= 1) return true;
                }
            }
        }
        return false;
    }

    int danger(Position p) const {
        if (risk_cache[index(p)] >= 0) return risk_cache[index(p)];
        int score = 0;
        for (auto e : enemies) {
            int visible_length = 0;
            bool partial = false;
            for (const auto &t : c.get_tiles())
                if (t.get_dragon() && t.get_dragon()->get_id() == e.get_id()) {
                    ++visible_length;
                    if (std::abs(delta(here.x, t.get_position().x, w)) == 3 ||
                        std::abs(delta(here.y, t.get_position().y, h)) == 3)
                        partial = true;
                }
            int budget = std::min(6, std::max(1, (partial ? std::max(visible_length, 4) : visible_length) - 1));

            static std::array<Position, BFS_Q> q;
            int lo = 0, hi = 0;
            q[hi++] = e.position;
            danger_depth[index(e.position)] = 0;
            int reach = INF;
            while (lo < hi) {
                auto cur = q[lo++];
                int n = danger_depth[index(cur)];
                if (cur == p) { reach = n; break; }
                if (n >= budget) continue;
                for (int d = 0; d < 4; ++d)
                    if (s.cells[index(cur)].edge[d] >= 0) {
                        auto next = destination(cur, d);
                        if (!next) continue;
                        auto to = *next;
                        auto tile = c.get_tile(to);
                        if (!tile) continue;
                        if (to == p) { reach = std::min(reach, n + 1); continue; }
                        if (tile->get_dragon() || danger_depth[index(to)] >= 0 || hi >= BFS_Q) continue;
                        danger_depth[index(to)] = n + 1;
                        if (hi < BFS_Q) q[hi++] = to;
                    }
            }
            for (int i = 0; i < hi; ++i) danger_depth[index(q[i])] = -1;
            if (reach < INF) score += 400 - 40 * reach;
            else if (dist(p, e.position) <= 3) score += 4;
        }
        risk_cache[index(p)] = score;
        return score;
    }

    int space(Position start) const {
        static std::array<Position, BFS_Q> q;
        int lo = 0, hi = 0;
        q[hi++] = start;
        space_seen[index(start)] = true;
        while (lo < hi) {
            auto p = q[lo++];
            for (int d = 0; d < 4; ++d)
                if (s.cells[index(p)].edge[d] == 0) {
                    auto n = step(p, d);
                    if (!space_seen[index(n)] && empty(n) && hi < BFS_Q) {
                        space_seen[index(n)] = true;
                        if (hi < BFS_Q) q[hi++] = n;
                    }
                }
        }
        for (int i = 0; i < hi; ++i) space_seen[index(q[i])] = false;
        return hi;
    }

    bool incoming() const {
        for (auto e : enemies)
            if (dist(e.position, here) <= 4 &&
                dist(step(e.position, di(e.get_dir())), here) < dist(e.position, here))
                return true;
        return false;
    }

    std::vector<Direction> route(Position target) const {
        std::vector<Direction> out;
        int at = index(target);
        while (at != index(here) && at >= 0 && arrival[at] >= 0) {
            out.push_back(DIRS[arrival[at]]);
            at = predecessor[at];
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    bool is_step_safe(int d, bool allow_enemy_head = false) const {
        int opp_dir = (di(c.get_dir()) + 2) % 4;
        if (d == opp_dir) return false;

        int edge = s.cells[index(here)].edge[d];
        if (edge < 0) return false;

        if (edge > 0) {
            if (s.resident || (portal_occupied(edge - 1) && !s.evacuating)) return false;
            auto exit = destination(here, d);
            if (exit) {
                auto t = c.get_tile(*exit);
                if (t) {
                    auto dragon = t->get_dragon();
                    if (dragon) {
                        if (dragon->get_id() == c.get_id()) return false;
                        if (dragon->get_team() == c.get_team()) return false;
                        if (!dragon->is_head()) return false;
                        if (!allow_enemy_head) return false;
                    }
                }
            }
            return true;
        }

        Position n = step(here, d);
        auto t = c.get_tile(n);
        if (!t) return false;
        auto dragon = t->get_dragon();
        if (dragon) {
            if (dragon->get_id() == c.get_id()) return false;
            if (dragon->get_team() == c.get_team()) return false;
            if (!dragon->is_head()) return false;
            if (!allow_enemy_head) return false;
        }
        return true;
    }

    std::optional<Action> split_kamikaze_action() {
        int my_len = c.get_length();
        if (my_len <= 4 || !c.can_split(2) || round - s.last_combat_split <= 8) return {};

        for (const auto &e : enemies) {
            int elen = enemy_effective_length(e.get_id());
            bool enemy_longer = (elen > my_len);

            // 1. Tail-Child Instant Same-Round Kill (spawns length-2 or length-3 child at tail_pos that kills e this round!)
            if (s.tail_visible && (enemy_longer || elen >= 4 || e.get_id() <= 1)) {
                for (int d = 0; d < 4; ++d) {
                    if (d == s.tail_facing) continue;
                    if (s.cells[index(s.tail_pos)].edge[d] == 0 && step(s.tail_pos, d) == e.position && c.can_split(2)) {
                        s.last_combat_split = round;
                        s.harvester_until = round + 10;
                        s.harvest_anchor = e.position;
                        return Action{{}, 2, Mode::Split, false};
                    }
                }
                if (my_len >= 5 && c.can_split(3)) {
                    for (int d1 = 0; d1 < 4; ++d1) {
                        if (d1 == s.tail_facing || s.cells[index(s.tail_pos)].edge[d1] != 0) continue;
                        Position mid = step(s.tail_pos, d1);
                        if (!empty(mid)) continue;
                        for (int d2 = 0; d2 < 4; ++d2) {
                            if (d2 == (d1 + 2) % 4 || s.cells[index(mid)].edge[d2] != 0) continue;
                            if (step(mid, d2) == e.position) {
                                s.last_combat_split = round;
                                s.harvester_until = round + 10;
                                s.harvest_anchor = e.position;
                                return Action{{}, 3, Mode::Split, false};
                            }
                        }
                    }
                }
            }

            // 2. Combat Missile Split (parent keeps my_len - 2..3 segments + full memory to harvest pearls, child spawns with 2..3 segments and kamikaze_lock!)
            if (round > s.born && enemy_longer && (!s.alpha || my_len >= 6)) {
                int d_head = distance[index(e.position)];
                int ed = di(e.get_dir());
                Position e_next = step(e.position, ed);
                if (d_head <= 3 && (d_head == 1 || dist(e_next, here) < dist(e.position, here))) {
                    int missile_len = (my_len >= 6 && d_head >= 2 && c.can_split(3)) ? 3 : 2;
                    if (c.can_split(missile_len)) {
                        s.last_combat_split = round;
                        s.harvester_until = round + 10;
                        s.harvest_anchor = e.position;
                        return Action{{}, missile_len, Mode::Split, false};
                    }
                }
            }
        }
        return {};
    }

    std::optional<Action> guaranteed_kill(bool high_value_only = false, bool safe_only = false) const {
        if (s.alpha) return {};
        if (c.get_unit_count() <= 1) return {};

        int my_len = c.get_length();
        int opp_dir = (di(c.get_dir()) + 2) % 4;
        bool locked_kam = (s.kamikaze_lock && round <= s.kamikaze_lock_until);

        auto is_longer_enemy = [&](int elen, int eid) -> bool {
            if (locked_kam && (elen > my_len || eid <= 1 || elen >= 4)) return true;
            return c.get_unit_count() >= 4 && elen > my_len;
        };

        auto trade_ok = [&](int elen, int eid) -> bool {
            if (is_longer_enemy(elen, eid)) return true;
            if (eid <= 1 || elen >= 4) return true;
            if (high_value_only) return false;
            if (elen >= my_len) return true;
            if (my_len <= 2) return true;
            return false;
        };

        // 1. Immediate 1-step collision (Kamikaze takes all valid trades; Neutral bots take all strictly longer enemies or length>=4 carries)
        for (const auto &e : enemies) {
            int elen = enemy_effective_length(e.get_id());
            bool longer = is_longer_enemy(elen, e.get_id());
            if (safe_only && !longer && !(my_len == 2 && c.get_unit_count() >= 4 && (elen >= 4 || (e.get_id() <= 1 && elen >= 3))))
                continue;
            if (!trade_ok(elen, e.get_id())) continue;
            for (int d = 0; d < 4; ++d) {
                if (d == opp_dir) continue;
                auto dest = destination(here, d);
                if (dest && *dest == e.position && is_step_safe(d, true))
                    return Action{{DIRS[d]}, 0, Mode::Kill, true};
            }
        }

        // 2. Sprint kill (2..5 steps onto a stationary enemy head; Neutral bots sprint-kill any strictly longer enemy!)
        int max_sprint = std::min(5, std::max(1, my_len - 1));
        for (int k = 2; k <= max_sprint; ++k) {
            for (const auto &e : enemies) {
                int elen = enemy_effective_length(e.get_id());
                bool longer = is_longer_enemy(elen, e.get_id());
                if (safe_only && !longer) continue;
                if (!trade_ok(elen, e.get_id())) continue;
                if (distance[index(e.position)] != k) continue;
                auto candidate = route(e.position);
                if (static_cast<int>(candidate.size()) != k) continue;
                if (di(candidate[0]) == opp_dir) continue;

                bool clear_path = true;
                Position cur = here;
                int prev_d = di(c.get_dir());
                for (int i = 0; i < k - 1; ++i) {
                    int step_d = di(candidate[i]);
                    if (step_d == (prev_d + 2) % 4) { clear_path = false; break; }
                    cur = step(cur, step_d);
                    if (!empty(cur)) { clear_path = false; break; }
                    prev_d = step_d;
                }
                if (clear_path && di(candidate[k - 1]) != (prev_d + 2) % 4)
                    return Action{candidate, 0, Mode::Kill, true};
            }
        }

        // 3. True 1-Exit Corridor trap (1-step works for length >= 2; 2-step sprint works for length >= 3)
        for (const auto &e : enemies) {
            int elen = enemy_effective_length(e.get_id());
            if (high_value_only && e.get_id() > 1 && elen < 4 && !is_longer_enemy(elen, e.get_id())) continue;
            int e_opp = (di(e.get_dir()) + 2) % 4;
            Position forced_tile = e.position;
            int enemy_escapes = 0;
            for (int ed = 0; ed < 4; ++ed) {
                if (ed == e_opp) continue;
                int edge = s.cells[index(e.position)].edge[ed];
                if (edge < 0) continue;
                if (edge == 0) {
                    Position ep = step(e.position, ed);
                    if (ep == here || empty(ep)) { ++enemy_escapes; forced_tile = ep; }
                } else ++enemy_escapes;
            }
            if (enemy_escapes == 1 && forced_tile != here) {
                if (distance[index(forced_tile)] == 1) {
                    int d = first[index(forced_tile)];
                    if (d >= 0 && d != opp_dir && is_step_safe(d, false))
                        return Action{{DIRS[d]}, 0, Mode::Kill, false};
                } else if (my_len >= 3 && distance[index(forced_tile)] == 2) {
                    auto candidate = route(forced_tile);
                    if (candidate.size() == 2) {
                        int d1 = di(candidate[0]);
                        int d2 = di(candidate[1]);
                        if (d1 != opp_dir && d2 != (d1 + 2) % 4 &&
                            s.cells[index(here)].edge[d1] == 0 &&
                            empty(step(here, d1)))
                            return Action{candidate, 0, Mode::Kill, false};
                    }
                }
            }
        }

        return {};
    }

    bool claimed(Position p) const {
        int my_dist = dist(here, p);
        for (auto f : friends) {
            int k = dist(f.position, p);
            if (!s.alpha && is_alpha(f.get_id()) && k <= 3 && round > threshold()) return true;
            if (k < my_dist || (k == my_dist && f.get_id() < c.get_id())) return true;
        }
        return false;
    }

    std::optional<Position> food_target(bool future) const {
        bool small_map_splitting_alpha = (s.alpha && map_scale() < 20 && !s.growing);
        bool kam = is_kamikaze();
        std::optional<Position> best;
        double bestscore = -1e20;
        std::vector<Position> pearls;
        for (const auto &t : c.get_tiles())
            if (t.has_pearl()) pearls.push_back(t.get_position());

        for (const auto &t : c.get_tiles()) {
            auto p = t.get_position();
            int n = distance[index(p)];
            if (n < 1 || n > 10 || t.get_dragon()) continue;
            if (future ? !(t.get_pearl_time() >= 1 && t.get_pearl_time() <= 2) : !t.has_pearl()) continue;
            if (claimed(p)) continue;

            int arr_d = arrival[index(p)];
            int first_d = first[index(p)];
            int exits = escape_count(p);
            int f_exits = forward_escape_count(p, arr_d);
            if (exits == 0 || f_exits == 0) continue;

            // Neutral and Alpha bots will not pursue a pearl if entering it or taking the first step toward it is certain death
            if (!kam) {
                if (is_dead_end_trap(p, arr_d)) continue;
                if (first_d >= 0) {
                    Position step1 = step(here, first_d);
                    if (is_dead_end_trap(step1, first_d) || is_enemy_certain_death(step1, first_d)) continue;
                }
                if (n == 1 && is_enemy_certain_death(p, arr_d)) continue;
            }

            int pearl_risk = s.alpha ? danger(p) : 0;
            if (s.alpha && !small_map_splitting_alpha) {
                if (pearl_risk >= 360) continue;
                if (pearl_risk >= 320 && (f_exits < 2 || n > 1)) continue;
                if (c.get_length() >= 12 && space(p) < 5) continue;
            }

            double cluster = 0;
            for (const auto &q : pearls) {
                int k = chebyshev(p, q);
                if (k > 0 && k <= 2) cluster += 2.0 / k;
            }

            double risk_weight = (s.alpha && !small_map_splitting_alpha) ? (pearl_risk >= 280 ? 0.06 : 0.02) : 0.0;
            double center_bonus = (!s.alpha || small_map_splitting_alpha) ? 0.35 * ((w + h) / 2 - center_dist(p)) : 0.0;
            double score = (45.0 + 18.0 * cluster) / (n + 0.5) - risk_weight * pearl_risk + center_bonus;

            if (!future && t.has_pearl()) score += 18.0 / (n + 1.0);
            if (future && n < t.get_pearl_time()) score -= 10;

            if (!best || score > bestscore || (score == bestscore && index(p) < index(*best))) {
                best = p;
                bestscore = score;
            }
        }
        return best;
    }

    struct PortalPlan {
        Position approach;
        int direction, id, cost;
    };

    std::optional<PortalPlan> portal_target(bool force) const {
        std::optional<PortalPlan> best;
        for (const auto &t : c.get_tiles())
            for (int d = 0; d < 4; ++d) {
                auto p = t.get_position();
                int e = s.cells[index(p)].edge[d];
                if (e <= 0 || distance[index(p)] == INF || (p != here && t.get_dragon())) continue;
                if (!force && !s.evacuating && portal_occupied(e - 1)) continue;
                if (p == here && d == (di(c.get_dir()) + 2) % 4) continue;
                if (s.resident) continue;
                auto exit = destination(p, d);
                if (exit) {
                    auto tile = c.get_tile(*exit);
                    if (tile && tile->get_dragon()) continue;
                }
                int cost = distance[index(p)] + 1;
                if (!best || cost < best->cost || (cost == best->cost && e - 1 < best->id))
                    best = PortalPlan{p, d, e - 1, cost};
            }
        return best;
    }

    std::optional<Position> feed_target() const {
        if (s.alpha) {
            if (auto sup = get_superior_alpha()) {
                for (const auto &f : friends)
                    if ((f.get_id() & SONAR_ID_MASK) == sup->id) return f.position;
                return sup->p;
            }
            return {};
        }
        std::optional<Position> target;
        int best_score = INF;
        for (const auto &a : s.alphas) {
            if (round - a.seen > SONAR_TTL) continue;
            int k = dist(here, a.p);
            int eff = k - std::min(8, (a.len - 6) / 2);
            if (eff < best_score) { best_score = eff; target = a.p; }
        }
        for (const auto &f : friends) {
            if (!is_alpha(f.get_id())) continue;
            int k = dist(here, f.position);
            int eff = k - std::min(8, (friendly_visible_length(f.get_id()) - 6) / 2);
            if (eff <= best_score) { best_score = eff; target = f.position; }
        }
        return target;
    }

    std::optional<Position> exploration_target() const {
        std::optional<Position> best;
        double best_score = -1e20;
        for (const auto &t : c.get_tiles()) {
            auto p = t.get_position();
            int path = distance[index(p)];
            if (path < 1 || path >= INF || t.get_dragon()) continue;
            int exits = escape_count(p);
            if (!exits) continue;
            if (is_dead_end_trap(p, arrival[index(p)])) continue;
            int age = std::min(80, round - s.cells[index(p)].visited);
            int unseen = 0;
            for (int d = 0; d < 4; ++d)
                if (s.cells[index(p)].edge[d] == 0 && s.cells[index(step(p, d))].seen < 0) ++unseen;
            double score = 0.7 * age + 16 * unseen + 5 * std::min(exits, 3) - 2 * path;
            score += 3 * (dist(here, s.sector_target) - dist(p, s.sector_target));
            if (s.alpha) score -= 0.10 * danger(p);
            for (const auto &f : friends) score -= 6 * std::max(0, 4 - dist(p, f.position));
            if (score > best_score) { best_score = score; best = p; }
        }
        return best;
    }

    std::optional<Position> remembered_pearl_target() const {
        int mscale = map_scale();
        if (mscale > 26 && round > s.harvester_until) return {};
        bool small_map_splitting_alpha = (s.alpha && mscale < 20 && !s.growing);
        std::optional<Position> best;
        double best_score = -1e20;
        for (int dy = -8; dy <= 8; ++dy) {
            int rem = 8 - std::abs(dy);
            for (int dx = -rem; dx <= rem; ++dx) {
                if (std::abs(dx) <= 3 && std::abs(dy) <= 3) continue;
                int d = std::abs(dx) + std::abs(dy);
                if (d < 4 || d > 8) continue;
                Position p = wrap({here.x + dx, here.y + dy});
                const auto &cell = s.cells[index(p)];
                if (cell.seen < 0 || cell.seen == round || cell.pearl_round < 0) continue;
                if (round - cell.seen > (mscale > 26 ? 10 : 24)) continue;
                if (cell.pearl_round > round + d) continue;
                int open_edges = 0;
                for (int ed = 0; ed < 4; ++ed)
                    if (cell.edge[ed] >= 0 || cell.edge[ed] == -2) ++open_edges;
                if (open_edges <= 1) continue;
                if (claimed(p)) continue;
                if (s.alpha && !small_map_splitting_alpha && round < 415 && danger(p) >= 320) continue;
                double score = 40.0 / (d + 1.0) - 0.35 * (round - cell.seen);
                if (!best || score > best_score) {
                    best = p;
                    best_score = score;
                }
            }
        }
        return best;
    }

    int remembered_direction(Position target) const {
        bool use_wall_memory = (map_scale() <= 26 || round <= s.harvester_until);
        int max_expand = use_wall_memory ? 200 : MAX_CELLS;
        int lo = 0, hi = 0;
        remembered_q[hi++] = here;
        remembered_first[index(here)] = 4;
        int best_frontier_dir = -1;
        int best_frontier_dist = INF;

        auto cleanup = [&]() {
            for (int i = 0; i < hi; ++i) remembered_first[index(remembered_q[i])] = -1;
        };

        while (lo < hi) {
            auto p = remembered_q[lo++];
            for (int d = 0; d < 4; ++d)
                if (s.cells[index(p)].edge[d] == 0) {
                    if (p == here && d == (di(c.get_dir()) + 2) % 4) continue;
                    auto n = step(p, d);
                    if (use_wall_memory) {
                        if (s.cells[index(n)].seen == round && s.cells[index(n)].has_dragon && n != target) continue;
                    } else {
                        if (!empty(n) && n != target) continue;
                    }
                    int first_d = (p == here) ? d : remembered_first[index(p)];
                    if (n == target) {
                        cleanup();
                        return first_d;
                    }
                    if (s.cells[index(n)].seen < 0) {
                        int fd = dist(n, target);
                        if (fd < best_frontier_dist) { best_frontier_dist = fd; best_frontier_dir = first_d; }
                        continue;
                    }
                    if (remembered_first[index(n)] >= 0) continue;
                    if (hi < max_expand) {
                        remembered_first[index(n)] = first_d;
                        remembered_q[hi++] = n;
                    }
                    int fd = dist(n, target);
                    if (fd < best_frontier_dist) { best_frontier_dist = fd; best_frontier_dir = first_d; }
                }
        }
        cleanup();
        return best_frontier_dir;
    }

    Action decide() {
        observe();
        paths();

        int opp_dir = (di(c.get_dir()) + 2) % 4;
        int feed_round = 415;

        // Alpha-for-Alpha Endgame Sacrifice:
        // If we have 2-3 long Alphas in the endgame, subordinate Alphas sacrifice into our singular Superior Alpha!
        int mscale = map_scale();
        bool receiver_alpha = s.alpha;
        auto sup_alpha = s.alpha ? get_superior_alpha() : std::optional<AlphaTrack>{};
        if (s.alpha && sup_alpha.has_value()) {
            int d_sup = dist(here, sup_alpha->p);
            int walk_est = (d_sup * 3) / 2;
            int needed_rounds = walk_est + std::min(16, c.get_length()) + 8;
            int min_merge_round = (mscale > 35) ? 385 : 405;
            int alpha_merge_round = std::clamp(500 - needed_rounds, min_merge_round, 425);
            bool close_growing_merge = (round >= 355 && d_sup <= 7 && danger(here) == 0 && c.get_unit_count() >= 3);
            if ((close_growing_merge || round >= alpha_merge_round) && walk_est + 6 <= (500 - round) &&
                (c.get_unit_count() > 2 || (c.get_unit_count() == 2 && d_sup <= 12))) {
                receiver_alpha = false;
            }
        }

        auto alpha = receiver_alpha ? std::optional<Position>{} : feed_target();
        bool feed = !receiver_alpha && (round >= feed_round || s.alpha) && alpha.has_value() &&
                    dist(here, *alpha) <= (500 - round) - 2;
        bool kam = is_kamikaze();
        bool sprint_hunter = is_sprint_hunter();
        bool opportunistic_kam = !s.alpha && !feed && has_longer_visible_enemy();

        // Protective Suicide: if a longer friendly Alpha/carry is cornered (escape_count <= 1) and we block or crowd it, sacrifice!
        if ((!s.alpha && c.get_length() <= 3 && c.get_unit_count() >= std::max(12, kamikaze_threshold() + 4)) ||
            (s.alpha && !receiver_alpha && c.get_unit_count() >= 2)) {
            for (const auto &f : friends) {
                int flen = friendly_visible_length(f.get_id());
                if (!s.alpha && flen < 6) continue;
                if (s.alpha && flen < c.get_length() && !(sup_alpha && (f.get_id() & SONAR_ID_MASK) == sup_alpha->id)) continue;
                int f_exits = escape_count(f.position);
                if (f_exits > 1) continue;
                Position f_front = step(f.position, di(f.get_dir()));
                auto t_front = c.get_tile(f_front);
                bool body_blocks_front = (t_front && t_front->get_dragon() && t_front->get_dragon()->get_id() == c.get_id());
                if (body_blocks_front || here == f_front)
                    return {{DIRS[opp_dir]}, 0, Mode::Feed, true};
                if (dist(here, f.position) == 1) {
                    int crowding_allies = 0;
                    for (const auto &other : friends)
                        if (other.get_id() != f.get_id() && dist(other.position, f.position) <= 2)
                            ++crowding_allies;
                    if (crowding_allies >= 1 || s.alpha)
                        return {{DIRS[opp_dir]}, 0, Mode::Feed, true};
                }
            }
        }

        // Split-and-Kamikaze: if we have length > 4 and spot an opponent bot longer than us, split off a length-2/3 kamikaze missile and feed on the pearls of both dragons!
        if (auto split_kam = split_kamikaze_action()) {
            return *split_kam;
        }

        if (!s.alpha && !s.growing && !feed && c.get_length() >= 4 && c.can_split(2))
            return {{}, 2, Mode::Split, false};

        // New children claim nearby portals before foraging or combat.
        if (!s.alpha && !s.resident && !s.kamikaze_lock && (s.evacuating || round - s.born < 8)) {
            auto early = portal_target(false);
            if (early) {
                int d = early->approach == here ? early->direction : first[index(early->approach)];
                if (d >= 0 && is_step_safe(d)) return {{DIRS[d]}, 0, Mode::Portal, false};
            }
        }

        // Feeder delivery: suicide within 3x3 (Chebyshev <= 2) of the Alpha so dropped pearls land within Alpha vision!
        if (feed && (c.get_unit_count() > 2 || (s.alpha && c.get_unit_count() == 2))) {
            for (const auto &f : friends) {
                int gap = dist(here, f.position);
                int c_gap = chebyshev(here, f.position);
                if (!is_alpha(f.get_id())) continue;
                if (s.alpha ? (c_gap > 3 || gap > 4) : (c_gap > 2 || gap > 3)) continue;
                if (s.alpha && (500 - round < 4)) continue;
                if (s.alpha && sup_alpha && (f.get_id() & SONAR_ID_MASK) != sup_alpha->id &&
                    friendly_visible_length(f.get_id()) < c.get_length())
                    continue;
                Position f_front = step(f.position, di(f.get_dir()));
                if (gap > 2 && round < 490) {
                    for (int d = 0; d < 4; ++d) {
                        if (s.cells[index(here)].edge[d] != 0 || !is_step_safe(d)) continue;
                        auto n = step(here, d);
                        if (n == f_front) continue;
                        auto tile = c.get_tile(n);
                        if (tile && !tile->has_pearl() && danger(n) == 0 && dist(n, f.position) < gap)
                            return {{DIRS[d]}, 0, Mode::Feed, false};
                    }
                }
                return {{DIRS[opp_dir]}, 0, Mode::Feed, true};
            }
        }

        // 0. High-value kill — for Kamikaze, Sprint Hunters, or Neutral bots that spot an enemy longer than them!
        if (kam || sprint_hunter || opportunistic_kam) {
            if (auto hv_kill = guaranteed_kill(true, false)) return *hv_kill;
        }

        bool medium_map = (mscale >= 20 && mscale <= 35);
        bool small_map_splitting_alpha = (s.alpha && mscale < 20 && !s.growing);

        // 1. Splitting
        if (!s.growing && !feed && c.get_length() >= 4 && c.can_split(2)) {
            return {{}, 2, Mode::Split, false};
        }

        auto pearls = food_target(false);
        bool adjacent_split_pearl = (!s.alpha && c.get_length() == 3 && pearls && distance[index(*pearls)] == 1);

        // 2. Standard kill — Kamikazes and Opportunistic Neutral Kamikazes (vs longer enemies) take kills!
        if (!s.alpha && !feed) {
            if (kam || (opportunistic_kam && !adjacent_split_pearl) || (sprint_hunter && !pearls)) {
                if (auto kill = guaranteed_kill(false, false)) return *kill;
            } else if (c.get_unit_count() >= 3) {
                if (auto safe_kill = guaranteed_kill(false, true)) return *safe_kill;
            }
        }

        bool allow_enemy_head = kam || (opportunistic_kam && !adjacent_split_pearl);
        std::vector<int> legal;
        for (int d = 0; d < 4; ++d)
            if (is_step_safe(d, allow_enemy_head)) legal.push_back(d);

        bool threatened = (s.alpha && !small_map_splitting_alpha) &&
                          ((incoming() && danger(here) >= 240) || danger(here) >= 280),
             perpendicular = false;
        bool has_safe_escape = false;
        for (int d : legal) {
            if (danger(step(here, d)) < 240) has_safe_escape = true;
            if ((d % 2) != (di(c.get_dir()) % 2) && danger(step(here, d)) < 240)
                perpendicular = true;
        }

        if (medium_map) {
            if (s.born < round && c.can_split(c.get_length() - 2) &&
                (legal.empty() || (s.alpha && (incoming() || danger(here) >= 320) && !has_safe_escape && !perpendicular)))
                return {{}, c.get_length() - 2, Mode::Split, false};
        } else {
            bool perp_safe = false;
            for (int d : legal)
                if ((d % 2) != (di(c.get_dir()) % 2) && danger(step(here, d)) < 240)
                    perp_safe = true;
            if (c.can_split(c.get_length() - 2) && (legal.empty() || (threatened && !perp_safe)))
                return {{}, c.get_length() - 2, Mode::Split, false};
        }

        auto future = pearls ? std::optional<Position>{} : food_target(true);
        auto rem_pearl = (!pearls && !future && !feed) ? remembered_pearl_target() : std::optional<Position>{};
        auto port = (!s.alpha || small_map_splitting_alpha) ? portal_target(false) : std::optional<PortalPlan>{};

        if (feed && pearls && alpha && dist(*pearls, *alpha) <= 5) pearls = std::nullopt;

        bool early_portal = (!s.alpha || small_map_splitting_alpha) && !s.kamikaze_lock &&
                            (s.evacuating || round - s.born < 8) && !s.resident;
        bool use_portal = port && ((!pearls && !future && !rem_pearl) || early_portal);

        std::optional<Position> target = pearls ? pearls : (future ? future : rem_pearl);
        Mode mode = s.resident ? Mode::Reside : Mode::Forage;

        if (!target && round <= s.harvester_until && dist(here, s.harvest_anchor) >= 1 && dist(here, s.harvest_anchor) <= 7) {
            target = s.harvest_anchor;
            mode = Mode::Forage;
        }

        if (feed) {
            bool grab_adjacent_pearl = pearls && (s.alpha || c.get_length() % 2 == 1) && distance[index(*pearls)] == 1 &&
                                       (!alpha || dist(*pearls, *alpha) > 5);
            if (!grab_adjacent_pearl) {
                target = alpha;
                mode = Mode::Feed;
            }
            use_portal = false;
        }
        if (use_portal) {
            mode = Mode::Portal;
            if (port->approach == here) {
                if (is_step_safe(port->direction, allow_enemy_head)) {
                    s.pending_portal = port->id;
                    occupy(port->id);
                    return {{DIRS[port->direction]}, 0, mode, false};
                }
            } else target = port->approach;
        }
        if (threatened) mode = Mode::Escape;

        // During endgame (round >= 405), Receiver Alpha stays near approaching feeders/subordinate Alphas
        if (receiver_alpha && round >= 405 && !target && !threatened) {
            int best_f = INF;
            for (const auto &f : friends) {
                int df = dist(here, f.position);
                if (df >= 2 && df < best_f && (round >= feed_round || is_alpha(f.get_id()))) {
                    best_f = df;
                    target = f.position;
                }
            }
        }

        bool patrol_only = false;
        if (!target && !feed && !threatened && !(receiver_alpha && round >= feed_round && !friends.empty()) &&
            dist(here, s.sector_target) > 2) {
            target = exploration_target();
            if (!target) target = s.sector_target;
            mode = s.dispersing ? Mode::Disperse : Mode::Forage;
            patrol_only = true;
        }

        // Ambush — Kamikaze bots and Neutral bots intercepting an approaching longer opponent bot
        if ((kam || (opportunistic_kam && !adjacent_split_pearl)) && !feed && !use_portal) {
            int my_len = c.get_length();
            for (auto e : enemies) {
                int elen = enemy_effective_length(e.get_id());
                bool longer = (elen > my_len) || s.kamikaze_lock;
                if (!kam && !longer) continue;
                bool is_enemy_alpha = (e.get_id() <= 1 || elen >= 4 || longer);
                if (pearls && !is_enemy_alpha && !s.kamikaze_lock) continue;

                int ed = di(e.get_dir());
                if (!s.kamikaze_lock && dist(step(e.position, ed), here) >= dist(e.position, here)) continue;
                if (!s.kamikaze_lock && e.get_dir() == c.get_dir()) continue;
                if (s.cells[index(e.position)].edge[ed] < 0) continue;

                auto aim = step(e.position, ed);
                if (empty(aim) && distance[index(aim)] < INF) { target = aim; mode = Mode::Ambush; patrol_only = false; break; }
                if (s.kamikaze_lock && distance[index(e.position)] < INF) { target = e.position; mode = Mode::Ambush; patrol_only = false; break; }
                auto lateral = step(aim, (ed + ((c.get_id() % 2) ? 1 : 3)) % 4);
                if (empty(lateral) && distance[index(lateral)] < INF) { target = lateral; mode = Mode::Ambush; patrol_only = false; break; }
            }
            if (kam && mode != Mode::Ambush && !pearls && round - s.enemy_alpha_seen <= 8 &&
                dist(here, s.enemy_alpha_pos) <= 9) {
                target = s.enemy_alpha_pos;
                mode = Mode::Ambush;
                patrol_only = false;
            }
        }

        int target_direction = target ? first[index(*target)] : -1;
        bool direct_feed = feed && target && target_direction >= 0;
        if (target && target_direction < 0) target_direction = remembered_direction(*target);

        // Direct BFS path for pearl/portal/visible-feed targets (avoiding pearls reserved for Alpha & friendly sole-exit traps)
        bool alpha_safe_fastpath = s.alpha && (pearls || future) && target_direction >= 0 &&
                                   danger(step(here, target_direction)) < 280;
        if ((!s.alpha || feed || small_map_splitting_alpha || alpha_safe_fastpath) && target &&
            (pearls || future || rem_pearl || use_portal || direct_feed) &&
            (use_portal || direct_feed || std::count(s.recent_path.begin(), s.recent_path.end(), here) < 3) &&
            target_direction >= 0 && is_step_safe(target_direction, allow_enemy_head)) {
            Position next_td = step(here, target_direction);
            auto tile_td = c.get_tile(next_td);
            bool steals_alpha_pearl = (feed && alpha && tile_td && tile_td->has_pearl() && dist(next_td, *alpha) <= 5);
            bool blocks_alpha_head = false;
            for (const auto &f : friends) {
                if (!s.alpha && (is_alpha(f.get_id()) || friendly_visible_length(f.get_id()) >= 6) &&
                    next_td == step(f.position, di(f.get_dir()))) {
                    blocks_alpha_head = true;
                    break;
                }
                if (dist(next_td, f.position) == 1 && escape_count(f.position) <= 1) {
                    blocks_alpha_head = true;
                    break;
                }
            }
            int edge_td = s.cells[index(here)].edge[target_direction];
            bool certain_death_step = !allow_enemy_head && edge_td == 0 &&
                                      (is_dead_end_trap(next_td, target_direction) || is_enemy_certain_death(next_td, target_direction));
            if (!steals_alpha_pearl && !blocks_alpha_head && !certain_death_step &&
                (edge_td > 0 || (escape_count(next_td) >= 1 && forward_escape_count(next_td, target_direction) >= 1)))
                return {{DIRS[target_direction]}, 0, mode, false};
        }

        double best = -1e30;
        int chosen = -1;
        int cur_dir = di(c.get_dir());

        for (int d : legal) {
            auto dest_opt = destination(here, d);
            if (!dest_opt) continue;
            auto n = *dest_opt;
            auto tile = c.get_tile(n);
            int exits = escape_count(n), f_exits = forward_escape_count(n, d), area = space(n);
            int risk = s.alpha ? danger(n) : 0;
            bool trap_step = !allow_enemy_head && s.cells[index(here)].edge[d] == 0 &&
                             (is_dead_end_trap(n, d) || is_enemy_certain_death(n, d));

            bool endgame_safe = (round >= 415 && risk == 0 && f_exits >= 1);
            bool has_food = !trap_step && !(early_portal && use_portal) && tile && tile->has_pearl() &&
                            (!feed || !alpha || dist(n, *alpha) > 5) &&
                            (!s.alpha || small_map_splitting_alpha || endgame_safe ||
                             (f_exits >= 1 && (risk < 280 || (risk < 320 && f_exits >= 2)) && (c.get_length() < 12 || area >= 5)));

            double risk_penalty = 0.0;
            if (s.alpha && !small_map_splitting_alpha) {
                if (risk >= 360)      risk_penalty = 520.0;
                else if (risk >= 320) risk_penalty = 240.0;
                else if (risk >= 280) risk_penalty = 100.0;
                else                  risk_penalty = 0.25 * risk;
            }
            double score = 2.0 * std::min(area, 10) + 4.0 * exits - risk_penalty;

            if (trap_step) score -= 950.0;
            if (exits == 0 || f_exits == 0) score -= 800;
            else if (exits == 1) {
                if (s.alpha && risk >= 280) score -= 80;
                else if (!has_food) score -= 15;
            }

            if (threatened && perpendicular && (d % 2) == (cur_dir % 2)) score -= 400;

            if (target) {
                double dir_bonus = patrol_only ? 28.0 : 80.0;
                double step_bonus = patrol_only ? 8.0 : 15.0;
                if (target_direction == d) score += dir_bonus;
                else score += step_bonus * (dist(here, *target) - dist(n, *target));
            }

            int cd_here = center_dist(here);
            int cd_next = center_dist(n);
            int pull_radius = std::max(3, (w + h) / 4);
            double pull_weight = 2.0;
            if (cd_next > pull_radius) score += pull_weight * (cd_here - cd_next);
            if ((n.x <= 2 || n.x >= w - 3) && (n.y <= 2 || n.y >= h - 3)) score -= 20.0;

            if (has_food) score += 120;
            else if (feed && alpha && tile && tile->has_pearl() && dist(n, *alpha) <= 5) score -= 350.0;

            if (tile && !tile->has_pearl() && tile->get_pearl_time() == 1) {
                if (distance[index(n)] <= 2 && !trap_step) score += 25;
                else score -= 15;
            }

            // Productive moves override revisit penalties; exploration and non-visible feeding use anti-cycling.
            bool bypass_anti_cycle = has_food || direct_feed || use_portal || s.resident ||
                                      (s.alpha && pearls.has_value() && target_direction == d);
            int v_time = s.cells[index(n)].visited;
            if (v_time < 0) score += 55.0;
            else if (!bypass_anti_cycle) {
                int v_age = round - v_time;
                if (v_age <= 2)        score -= 280.0;
                else if (v_age <= 6)   score -= 120.0;
                else if (v_age <= 20)  score -= 45.0;
                else if (v_age <= 55)  score -= 12.0;
                else                   score += std::min(v_age, 200) * 0.22;
            }

            if (!bypass_anti_cycle) {
                for (int i = static_cast<int>(s.recent_path.size()) - 1; i >= 0; --i) {
                    if (s.recent_path[i] == n) {
                        int age = static_cast<int>(s.recent_path.size() - i);
                        if (age <= 6) score -= 220.0;
                        else if (age <= 14) score -= 70.0;
                        break;
                    }
                }
            }

            Position ahead = n;
            for (int step_k = 1; step_k <= 3; ++step_k) {
                if (s.cells[index(ahead)].edge[d] != 0) break;
                ahead = step(ahead, d);
                if (s.cells[index(ahead)].seen < 0) score += 12.0;
                else if (s.cells[index(ahead)].visited < 0) score += 6.0;
            }

            if (d == cur_dir) score += 12.0;
            int turn_diff = (d - cur_dir + 4) % 4;
            if (turn_diff == 1 || turn_diff == 3) score -= 8.0;

            if (!s.alpha) {
                for (auto f : friends) {
                    int d_team = dist(n, f.position);
                    if (d_team == 1 && escape_count(f.position) <= 1) score -= 650.0;
                    if (is_alpha(f.get_id()) || friendly_visible_length(f.get_id()) >= 6) {
                        if (n == step(f.position, di(f.get_dir()))) score -= 350.0;
                        else if (!feed && d_team <= 2) score -= 65.0;
                    } else if (!feed && d_team <= 4) {
                        score -= 6.0 * (5 - d_team);
                    }
                }
            } else if (round < 415 && !feed) {
                for (auto f : friends) {
                    int d_team = dist(n, f.position);
                    if (d_team == 1 && escape_count(f.position) <= 1) score -= 650.0;
                    if (d_team <= 4) score -= 6.0 * (5 - d_team);
                }
            }

            if (score > best) { best = score; chosen = d; }
        }

        if (chosen >= 0) return {{DIRS[chosen]}, 0, mode, false};

        for (int d = 0; d < 4; ++d)
            if (is_step_safe(d, false)) {
                int edge = s.cells[index(here)].edge[d];
                if (edge > 0) {
                    occupy(edge - 1);
                    return {{DIRS[d]}, 0, Mode::Portal, false};
                }
            }

        auto forced = portal_target(true);
        if (forced && forced->approach == here && forced->direction != opp_dir) {
            occupy(forced->id);
            return {{DIRS[forced->direction]}, 0, Mode::Portal, false};
        }

        if (c.can_split(c.get_length() - 2)) return {{}, c.get_length() - 2, Mode::Split, false};

        for (int d = 0; d < 4; ++d)
            if (is_step_safe(d, true)) return {{DIRS[d]}, 0, Mode::Trapped, true};

        for (int d = 0; d < 4; ++d) {
            if (d == opp_dir || s.cells[index(here)].edge[d] < 0) continue;
            auto opt_dest = destination(here, d);
            if (opt_dest) {
                auto t = c.get_tile(*opt_dest);
                if (t && t->get_dragon() && t->get_dragon()->get_team() == c.get_team() &&
                    t->get_dragon()->is_head()) continue;
            }
            return {{DIRS[d]}, 0, Mode::Trapped, false};
        }

        for (int d = 0; d < 4; ++d) {
            auto opt_dest = destination(here, d);
            if (opt_dest) {
                auto t = c.get_tile(*opt_dest);
                if (t && t->get_dragon() && t->get_dragon()->get_team() == c.get_team() &&
                    t->get_dragon()->is_head()) continue;
            }
            return {{DIRS[d]}, 0, Mode::Trapped, true};
        }

        return {{c.get_dir()}, 0, Mode::Trapped, true};
    }

    void execute(const Action &a) {
        Position post = here;
        if (a.child) {
            c.do_split(a.child);
        } else {
            for (auto d : a.moves) {
                int e = s.cells[index(post)].edge[di(d)];
                auto n = destination(post, di(d));
                if (e > 0) { s.pending_portal = e - 1; occupy(e - 1); }
                if (!n) break;
                auto t = c.get_tile(*n);
                if (t && t->has_pearl()) s.last_food = round;
                post = *n;
            }

            if (a.moves.empty()) c.make_move(c.get_dir());
            else if (a.moves.size() == 1) c.make_move(a.moves.front());
            else c.make_moves(a.moves);
        }

        if (!a.intentional_death && (s.pending_portal < 0 ||
            (!a.moves.empty() && destination(here, di(a.moves.front())).has_value()))) {
            bool broadcast_self_alpha = s.alpha && (a.mode != Mode::Feed);
            if (broadcast_self_alpha) {
                std::uint64_t pkt = alpha_packet64(c.get_id(), post, round, c.get_length());
                for (int d = 0; d < 4; ++d) c.send_sonar(DIRS[d], pkt);
            } else {
                std::vector<AlphaTrack> active;
                for (const auto &ai : s.alphas)
                    if (round - ai.seen <= SONAR_TTL) active.push_back(ai);
                std::sort(active.begin(), active.end(), [&](const AlphaTrack &x, const AlphaTrack &y) {
                    if (x.len != y.len) return x.len > y.len;
                    int dx = dist(post, x.p), dy = dist(post, y.p);
                    return dx != dy ? dx < dy : x.seen > y.seen;
                });
                if (!active.empty()) {
                    for (int d = 0; d < 4; ++d) {
                        const auto &ai = active[d % active.size()];
                        c.send_sonar(DIRS[d], alpha_packet64(ai.id, ai.p, ai.seen, ai.len));
                    }
                } else if (round - s.enemy_alpha_seen <= 12 || is_kamikaze_regime()) {
                    std::uint64_t pkt = alpha_packet64(SONAR_ID_MASK, {0, 0}, round, 0);
                    for (int d = 0; d < 4; ++d) c.send_sonar(DIRS[d], pkt);
                }
            }
        }

        const char *role_prefix = s.alpha ? "Alpha:" : (is_kamikaze() ? "Kamikaze:" : "Neutral:");
        c.set_indicator_string(std::string(role_prefix) + name(a.mode));
    }
};

} // namespace bot

int main() {
    try {
        auto [ct, game] = unswbc::init();
        auto brain = std::make_unique<bot::Brain>(ct, game);

        while (unswbc::update(ct, game)) {
            try {
                auto action = brain->decide();
                brain->execute(action);
            } catch (...) {
                ct.make_move(ct.get_dir());
            }
            unswbc::end_turn();
        }
    } catch (...) { return 0; }
    return 0;
}
